#!/usr/bin/env python3
"""
Benchmark ResNet-50 training throughput with LanceDB as the data source.

This benchmark is designed to stress the data loader: ResNet-50 is fast
enough on a modern GPU that it is easily I/O bound when reading images from
cloud storage, making data pipeline efficiency the dominant variable.

Each log line breaks down time into:
  - img/s:      images processed per second (wall-clock)
  - gpu_ms:     time spent on forward + backward pass (GPU-side)
  - data_ms:    wall-clock time per step minus gpu_ms (CPU/IO-side)
  - u/r/c/d:    pipeline row counts: unscanned / raw / cooked / done
  - MB/s:       raw bytes fetched from storage per second
  - ld:         avg time per step waiting for LanceDB I/O (load)
  - xf:         avg time per step for JPEG decode + image transforms
  - cpu%:       system CPU utilization averaged over the log interval
  - vram%:      GPU VRAM in use as a percentage of total device memory

When data_ms >> gpu_ms the pipeline is I/O bound.
When gpu_ms >> data_ms the GPU is the bottleneck.
When fetch_ms dominates data_ms, S3 latency/bandwidth is the limit.
When xform_ms dominates data_ms, JPEG decode / image transforms are the limit.

Required environment variables:
    LANCEDB_URI      URI of the LanceDB database (e.g. s3://my-bucket/data)

Single GPU:
    python benchmark.py

8 GPUs:
    torchrun --nproc_per_node=8 benchmark.py
"""

import os
import sys
import logging
import time

try:
    import psutil as _psutil
    _psutil.cpu_percent()  # prime so first interval read is accurate
    _psutil_ok = True
except ImportError:
    _psutil_ok = False


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb", "python", "python"))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.io import decode_jpeg
import functools
import lancedb
from lancedb.streaming import StreamingDataset


def _open_table(uri: str, table_name: str):
    """Module-level factory so it survives pickling into DataLoader workers."""
    return lancedb.connect(uri).open_table(table_name)

# ── Configuration ──────────────────────────────────────────────────────────────
TABLE_NAME   = "resnet_images"
NUM_EPOCHS   = 5
NUM_SPLITS   = int(os.environ.get("NUM_SPLITS",   8))
NUM_WORKERS  = int(os.environ.get("NUM_WORKERS",  1))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", 512))
SHUFFLE_SEED = 42
LOG_INTERVAL = 20     # log every N steps

# StreamingDataset I/O tuning
READ_BATCH_SIZE  = int(os.environ.get("READ_BATCH_SIZE",  64))  # Rows fetched per split per take_offsets call.
PREFETCH_BATCHES = int(os.environ.get("PREFETCH_BATCHES",  1))  # Concurrent read_batch_size fetches in flight per split.


# ── Transforms ─────────────────────────────────────────────────────────────────

_crop = transforms.RandomResizedCrop(224)
_flip = transforms.RandomHorizontalFlip()
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


# ── Transform / Collate ────────────────────────────────────────────────────────

def decode_transform(batch):
    """Decode JPEG and apply image transforms in the background prefetch thread.

    Uses zero-copy access to Arrow's binary column buffers: the JPEG bytes
    are never copied to the Python heap.  The Arrow data buffer is viewed
    directly via numpy (zero-copy) and then handed to decode_jpeg via
    torch.from_numpy (also zero-copy).  decode_jpeg only reads its input,
    so the read-only Arrow memory is safe here.
    """
    col     = batch.column("image")
    bufs    = col.buffers()
    offsets = np.frombuffer(bufs[1], dtype=np.int64)  # int64 for large_binary
    data    = np.frombuffer(bufs[2], dtype=np.uint8)

    # Decode: each decode_jpeg call releases the GIL
    imgs = [
        decode_jpeg(torch.from_numpy(data[offsets[i] : offsets[i + 1]]))
        for i in range(len(col))
    ]

    # Per-image crop + flip so each gets independent random parameters
    imgs = [_flip(_crop(img)) for img in imgs]

    # Stack once, then cast + normalize as a single batched op
    batch_t = torch.stack(imgs).to(torch.float32).div_(255.0)
    batch_t.sub_(_MEAN).div_(_STD)

    labels = batch.column("label").to_pylist()
    return [{"image": batch_t[i], "label": lbl} for i, lbl in enumerate(labels)]


def collate_fn(rows):
    return (
        torch.stack([row["image"] for row in rows]),
        torch.tensor([row["label"] for row in rows], dtype=torch.long),
    )


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
    total_vram = torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else 0
    model = models.resnet50(weights=None).to(device, dtype=torch.bfloat16)
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
            transform=decode_transform,
            connection_factory=functools.partial(_open_table, lancedb_uri),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            multiprocessing_context="forkserver" if NUM_WORKERS > 0 else None,
            # pin_memory only helps when workers pre-pin batches while the GPU
            # runs the previous step.  With num_workers=0 everything is
            # synchronous in the main thread so pinning is in the critical
            # path regardless — no benefit.
            pin_memory=NUM_WORKERS > 0,
        )

        model.train()
        interval_images    = 0
        interval_gpu_ms    = 0.0
        interval_start     = time.perf_counter()
        prev_bytes_loaded  = dataset.bytes_loaded
        prev_fetch_time    = dataset.fetch_time
        prev_transform_time = dataset.transform_time

        for step, (images, labels) in enumerate(dataloader):
            images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
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

                interval_mb    = (dataset.bytes_loaded - prev_bytes_loaded) / 1e6
                mb_per_sec     = interval_mb / elapsed if elapsed > 0 else 0.0
                avg_fetch_ms   = (dataset.fetch_time - prev_fetch_time) * 1000 / LOG_INTERVAL
                avg_xform_ms   = (dataset.transform_time - prev_transform_time) * 1000 / LOG_INTERVAL
                unscanned    = dataset.unscanned_rows
                raw_depth    = dataset.raw_queue_depth
                cooked_depth = dataset.prefetch_queue_depth
                complete     = dataset.consumed_rows
                cpu_str = f"{_psutil.cpu_percent():.0f}%" if _psutil_ok else "n/a"
                if total_vram > 0:
                    vram_pct = torch.cuda.memory_allocated(device) / total_vram * 100
                    vram_str = f"{vram_pct:.0f}%"
                else:
                    vram_str = "n/a"

                logging.info(
                    f"epoch {epoch + 1}/{NUM_EPOCHS} "
                    f"step {step + 1:5d} | "
                    f"loss {loss.item():.4f} | "
                    f"{img_per_sec:,.0f} img/s | "
                    f"gpu {avg_gpu_ms:.1f}ms | "
                    f"data {avg_data_ms:.1f}ms | "
                    f"{unscanned}/{raw_depth}/{cooked_depth}/{complete} | "
                    f"{mb_per_sec:.1f} MB/s | "
                    f"ld {avg_fetch_ms:.1f}ms | "
                    f"xf {avg_xform_ms:.1f}ms | "
                    f"cpu {cpu_str} vram {vram_str}"
                )
                interval_images     = 0
                interval_gpu_ms     = 0.0
                interval_start      = time.perf_counter()
                prev_bytes_loaded   = dataset.bytes_loaded
                prev_fetch_time     = dataset.fetch_time
                prev_transform_time = dataset.transform_time

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
