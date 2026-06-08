#!/usr/bin/env python3
"""
Download the Alpaca dataset and store it in a LanceDB table.

Run once before training:
    /home/pace/venvs/lancedb/bin/python prepare_data.py
"""

import os
import sys

# Use lancedb from the submodule (requires the Rust extension to be built:
#   cd lancedb/python && maturin develop --release)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb", "python", "python"))

import pyarrow as pa
import lancedb
from datasets import load_dataset

DB_PATH = "./data"
TABLE_NAME = "alpaca"
# Must divide evenly into world_size * num_workers (e.g. 8 GPUs * 4 workers = 32).
# 64 works for: 1, 2, 4, 8, 16 GPUs with 1, 2, 4 workers each.
NUM_SPLITS = 64


def main():
    print("Downloading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    n_orig = len(dataset)

    # StreamingDataset requires num_rows % num_splits == 0
    n = (n_orig // NUM_SPLITS) * NUM_SPLITS
    dataset = dataset.select(range(n))
    print(f"Dataset: {n_orig} examples -> {n} (truncated to nearest multiple of {NUM_SPLITS})")

    table_data = pa.table({
        "instruction": pa.array(dataset["instruction"], type=pa.string()),
        "input": pa.array(dataset["input"], type=pa.string()),
        "output": pa.array(dataset["output"], type=pa.string()),
    })

    os.makedirs(DB_PATH, exist_ok=True)
    db = lancedb.connect(DB_PATH)

    if TABLE_NAME in db.table_names():
        print(f"Dropping existing '{TABLE_NAME}' table...")
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=table_data)
    print(f"Stored {len(table)} rows in '{DB_PATH}/{TABLE_NAME}'")
    print(f"Schema: {table.schema}")


if __name__ == "__main__":
    main()
