#!/usr/bin/env python3
"""
Fine-tune an LLM on the Alpaca dataset stored in LanceDB.

Prerequisites:
    1. Build lancedb from the submodule:
           cd lancedb/python && maturin develop --release && cd ../..
    2. Install dependencies:
           pip install transformers peft accelerate object-store-python
    3. Prepare data:
           LANCEDB_URI=s3://my-bucket/data python prepare_data.py

Required environment variables:
    LANCEDB_URI      URI of the LanceDB database  (e.g. s3://my-bucket/data)
    CHECKPOINT_URI   URI for checkpoint storage    (e.g. s3://my-bucket/checkpoints)

Single GPU:
    python train.py

8 GPUs:
    torchrun --nproc_per_node=8 train.py
"""

import io
import os
import sys
import logging
import time
import tempfile
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lancedb", "python", "python"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from object_store import ObjectStore
import lancedb
from lancedb.streaming import StreamingDataset

# ── Configuration ──────────────────────────────────────────────────────────────
# Ungated, fast to download. Swap for a larger model once throughput is confirmed:
#   "meta-llama/Llama-3.2-1B"           (requires HF token)
#   "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TABLE_NAME = "alpaca"

NUM_EPOCHS = 100
# Must divide evenly into world_size.
# 64 works for 1, 2, 4, 8, 16, or 32 GPUs.
NUM_SPLITS = 64
NUM_WORKERS = 0       # DataLoader workers per GPU. Start at 0; increase if I/O bound.
BATCH_SIZE = 4        # Per-GPU micro-batch size.
MAX_LENGTH = 512      # Truncate sequences to this many tokens.
SHUFFLE_SEED = 42
LEARNING_RATE = 2e-4
LOG_INTERVAL = 50     # Log throughput every N steps.

# StreamingDataset I/O tuning
READ_BATCH_SIZE  = 64   # Rows fetched per split per take_offsets call.
PREFETCH_BATCHES = 4    # Concurrent read_batch_size fetches in flight per split.

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]


# ── Distributed helpers ────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _parse_uri(uri: str):
    """Return (ObjectStore base URL, path prefix) from a full URI."""
    parsed = urlparse(uri)
    if parsed.scheme in ("s3", "gs", "az", "abfs"):
        base = f"{parsed.scheme}://{parsed.netloc}"
        prefix = parsed.path.lstrip("/")
    else:
        base = uri
        prefix = ""
    return base, prefix



def save_checkpoint(store: ObjectStore, prefix: str, name: str, obj) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    key = f"{prefix}/{name}" if prefix else name
    store.put(key, buf)


def save_adapter(store: ObjectStore, prefix: str, model, tokenizer) -> None:
    """Save a PEFT adapter to object storage via a temporary local directory."""
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        tokenizer.save_pretrained(tmp)
        for fname in os.listdir(tmp):
            key = f"{prefix}/final_adapter/{fname}" if prefix else f"final_adapter/{fname}"
            with open(os.path.join(tmp, fname), "rb") as f:
                store.put(key, f.read())


# ── Data helpers ───────────────────────────────────────────────────────────────

def format_example(row: dict) -> str:
    instruction = str(row.get("instruction", ""))
    inp = str(row.get("input", ""))
    output = str(row.get("output", ""))
    if inp.strip():
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        texts = [format_example(row) + tokenizer.eos_token for row in batch]
        encoded = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }
    return collate_fn


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rank, world_size = setup_distributed()
    is_main = rank == 0

    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Environment ───────────────────────────────────────────────────────────
    lancedb_uri = os.environ.get("LANCEDB_URI")
    if not lancedb_uri:
        print("Error: LANCEDB_URI environment variable is not set.")
        sys.exit(1)

    checkpoint_uri = os.environ.get("CHECKPOINT_URI")
    if not checkpoint_uri:
        print("Error: CHECKPOINT_URI environment variable is not set.")
        sys.exit(1)

    checkpoint_store_base, checkpoint_prefix = _parse_uri(checkpoint_uri)
    checkpoint_store = ObjectStore(checkpoint_store_base)

    if is_main:
        logging.info(f"World size: {world_size}")
        logging.info(f"Model: {MODEL_NAME}")
        logging.info(f"LanceDB: {lancedb_uri}")
        logging.info(f"Checkpoints: {checkpoint_uri}")

    # ── Model + tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    model = torch.compile(model, dynamic=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── LanceDB ───────────────────────────────────────────────────────────────
    db = lancedb.connect(lancedb_uri)
    table = db.open_table(TABLE_NAME)
    if is_main:
        logging.info(f"LanceDB table: {len(table)} rows, {NUM_SPLITS} splits")

    collate_fn = make_collate_fn(tokenizer)

    # ── Training loop ─────────────────────────────────────────────────────────
    total_tokens = 0
    total_steps = 0
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
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_tokens = 0
        epoch_rows = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_rows = input_ids.shape[0]
            batch_tokens = int(attention_mask.sum())
            epoch_rows += batch_rows
            epoch_tokens += batch_tokens
            total_tokens += batch_tokens
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1

            if is_main and total_steps % LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - training_start
                tok_per_sec = total_tokens / elapsed
                rows_per_sec = epoch_rows / (time.perf_counter() - epoch_start)
                logging.info(
                    f"epoch {epoch + 1:3d}/{NUM_EPOCHS} "
                    f"step {step + 1:5d} | "
                    f"loss {loss.item():.4f} | "
                    f"{tok_per_sec:,.0f} tok/s | "
                    f"{rows_per_sec:,.0f} rows/s"
                )

        epoch_time = time.perf_counter() - epoch_start
        if is_main:
            avg_loss = epoch_loss / epoch_steps if epoch_steps else 0
            logging.info(
                f"=== epoch {epoch + 1:3d} | "
                f"loss {avg_loss:.4f} | "
                f"{epoch_time:.1f}s | "
                f"{epoch_tokens:,} tokens"
            )

            # Save dataset state for resumability
            checkpoint = {
                "epoch": epoch,
                "dataset_state": dataset.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(
                checkpoint_store, checkpoint_prefix,
                f"checkpoint_epoch_{epoch + 1:03d}.pt", checkpoint,
            )

    if is_main:
        elapsed = time.perf_counter() - training_start
        logging.info(
            f"Training complete | "
            f"{elapsed / 60:.1f} min | "
            f"{total_tokens:,} total tokens | "
            f"{total_tokens / elapsed:,.0f} avg tok/s"
        )
        # Save final LoRA adapter
        base = model.module if world_size > 1 else model
        save_adapter(checkpoint_store, checkpoint_prefix, base, tokenizer)
        logging.info(f"Adapter saved to {checkpoint_uri}/final_adapter")

    cleanup_distributed()


if __name__ == "__main__":
    import multiprocessing
    import resource
    multiprocessing.set_start_method("forkserver")
    # torch.compile opens many files while tracing; raise the fd limit to avoid EMFILE.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    main()
