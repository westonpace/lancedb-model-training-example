#!/usr/bin/env python3
"""
Benchmark ResNet-50 training throughput with LanceDB as the data source.

This benchmark is designed to stress the data loader: ResNet-50 is fast
enough on a modern GPU that it is easily I/O bound when reading images from
cloud storage, making data pipeline efficiency the dominant variable.

Each log line breaks down time into:
  - data_ms:  time spent fetching + decoding images (CPU-side)
  - gpu_ms:   time spent on forward + backward pass (GPU-side)
  - img/s:    images processed per second (wall-clock)

When data_ms >> gpu_ms the pipeline is I/O bound.
When gpu_ms >> data_ms the GPU is the bottleneck.

Required environment variables:
    LANCEDB_URI      URI of the LanceDB database (e.g. s3://my-bucket/data)

Single GPU:
    python benchmark.py

8 GPUs:
    torchrun --nproc_per_node=8 benchmark.py
"""

import io
import os
import sys
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb", "python", "python"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import lancedb
from lancedb.streaming import StreamingDataset

# ── Configuration ──────────────────────────────────────────────────────────────
TABLE_NAME   = "resnet_images"
NUM_EPOCHS   = 5
NUM_SPLITS   = 64
NUM_WORKERS  = 0
BATCH_SIZE   = 64
SHUFFLE_SEED = 42
LOG_INTERVAL = 20     # log every N steps

# StreamingDataset I/O tuning
READ_BATCH_SIZE  = 64   # Rows fetched per split per take_offsets call.
PREFETCH_BATCHES = 4    # Concurrent read_batch_size fetches in flight per split.


# ── Transforms ─────────────────────────────────────────────────────────────────

_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Distributed helpers ────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Collate ────────────────────────────────────────────────────────────────────

def collate_fn(rows):
    images = []
    labels = []
    for row in rows:
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        images.append(_train_transform(img))
        labels.append(row["label"])
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rank, world_size = setup_distributed()
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    lancedb_uri = os.environ.get("LANCEDB_URI")
    if not lancedb_uri:
        print("Error: LANCEDB_URI environment variable is not set.")
        sys.exit(1)

    # ── Model ──────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None).to(device)
    model = torch.compile(model, dynamic=True)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=1e-4)

    if is_main:
        logging.info(f"World size: {world_size} | Device: {device}")
        logging.info(f"Batch size: {BATCH_SIZE} | Splits: {NUM_SPLITS}")

    # ── LanceDB ────────────────────────────────────────────────────────────────
    db = lancedb.connect(lancedb_uri)
    table = db.open_table(TABLE_NAME)
    if is_main:
        logging.info(f"Table: {len(table):,} rows")

    # ── CUDA timing events ─────────────────────────────────────────────────────
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    total_images = 0
    total_steps  = 0
    training_start = time.perf_counter()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()

        dataset = StreamingDataset(
            table,
            num_splits=NUM_SPLITS,
            shuffle_seed=SHUFFLE_SEED,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            read_batch_size=READ_BATCH_SIZE,
            prefetch_batches=PREFETCH_BATCHES,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            multiprocessing_context="forkserver" if NUM_WORKERS > 0 else None,
        )

        model.train()
        interval_images  = 0
        interval_gpu_ms  = 0.0
        interval_start   = time.perf_counter()

        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            start_event.record()
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_event.record()
            torch.cuda.synchronize()

            gpu_ms = start_event.elapsed_time(end_event)
            batch_n = images.shape[0]

            interval_images += batch_n
            total_images    += batch_n
            total_steps     += 1
            interval_gpu_ms += gpu_ms

            if is_main and (total_steps % LOG_INTERVAL == 0):
                elapsed     = time.perf_counter() - interval_start
                img_per_sec = interval_images / elapsed
                avg_gpu_ms  = interval_gpu_ms / LOG_INTERVAL
                avg_data_ms = elapsed * 1000 / LOG_INTERVAL - avg_gpu_ms

                logging.info(
                    f"epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"step {step + 1:5d} | "
                    f"loss {loss.item():.4f} | "
                    f"{img_per_sec:,.0f} img/s | "
                    f"gpu {avg_gpu_ms:.1f}ms | "
                    f"data {avg_data_ms:.1f}ms"
                )
                interval_images  = 0
                interval_gpu_ms  = 0.0
                interval_start   = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start
        if is_main:
            logging.info(
                f"=== epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"{epoch_time:.1f}s | "
                f"{total_images / (time.perf_counter() - training_start):,.0f} avg img/s"
            )

    cleanup_distributed()


if __name__ == "__main__":
    import multiprocessing
    import resource
    multiprocessing.set_start_method("forkserver")
    # torch.compile opens many files while tracing; raise the fd limit to avoid EMFILE.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    main()
