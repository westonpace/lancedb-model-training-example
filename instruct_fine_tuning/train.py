#!/usr/bin/env python3
"""
Fine-tune an LLM on the Alpaca dataset stored in LanceDB.

Prerequisites:
    1. Build lancedb from the submodule:
           cd lancedb/python && maturin develop --release && cd ../..
    2. Install dependencies:
           pip install transformers peft accelerate
    3. Prepare data:
           python prepare_data.py

Single GPU:
    python train.py

8 GPUs:
    torchrun --nproc_per_node=8 train.py
"""

import os
import sys
import logging
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb", "python", "python"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import lancedb
from lancedb.streaming import StreamingDataset

# ── Configuration ──────────────────────────────────────────────────────────────
# Ungated, fast to download. Swap for a larger model once throughput is confirmed:
#   "meta-llama/Llama-3.2-1B"           (requires HF token)
#   "microsoft/Phi-3-mini-4k-instruct"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DB_PATH = "./data"
TABLE_NAME = "alpaca"

NUM_EPOCHS = 100
# Must divide num_rows AND (world_size * NUM_WORKERS).
# 64 works for 8 GPUs with 1, 2, 4 or 8 DataLoader workers.
NUM_SPLITS = 64
NUM_WORKERS = 0       # DataLoader workers per GPU. Start at 0; increase if I/O bound.
BATCH_SIZE = 4        # Per-GPU micro-batch size.
MAX_LENGTH = 512      # Truncate sequences to this many tokens.
SHUFFLE_SEED = 42
LEARNING_RATE = 2e-4
CHECKPOINT_DIR = "./checkpoints"
LOG_INTERVAL = 50     # Log throughput every N steps.

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
            padding="max_length",
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

    if is_main:
        logging.info(f"World size: {world_size}")
        logging.info(f"Model: {MODEL_NAME}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── LanceDB ───────────────────────────────────────────────────────────────
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    if is_main:
        logging.info(f"LanceDB table: {len(table)} rows, {NUM_SPLITS} splits")

    collate_fn = make_collate_fn(tokenizer)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
        )
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
        )

        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_tokens = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_tokens = int(attention_mask.sum())
            epoch_tokens += batch_tokens
            total_tokens += batch_tokens
            epoch_loss += loss.item()
            epoch_steps += 1
            total_steps += 1

            if is_main and total_steps % LOG_INTERVAL == 0:
                elapsed = time.perf_counter() - training_start
                tok_per_sec = total_tokens / elapsed
                logging.info(
                    f"epoch {epoch + 1:3d}/{NUM_EPOCHS} "
                    f"step {step + 1:5d} | "
                    f"loss {loss.item():.4f} | "
                    f"{tok_per_sec:,.0f} tok/s"
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
            torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch + 1:03d}.pt")

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
        base.save_pretrained(f"{CHECKPOINT_DIR}/final_adapter")
        tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/final_adapter")
        logging.info(f"Adapter saved to {CHECKPOINT_DIR}/final_adapter")

    cleanup_distributed()


if __name__ == "__main__":
    main()
