# syntax=docker/dockerfile:1
# NGC PyTorch base includes CUDA, cuDNN, NCCL, and PyTorch pre-installed.
# 25.11 ships with CUDA 13.0.2, matching the maximum supported by driver 580.
# Upgrade this tag when the host driver is updated (25.12+ requires CUDA 13.1+).
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install Rust (required to compile the lancedb extension).
# Rust is only needed at build time; the compiled .so is what ships in the image.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin (Rust/Python build tool) and Python dependencies.
RUN pip install --no-cache-dir \
    maturin \
    transformers \
    peft \
    accelerate \
    datasets \
    pyarrow \
    object-store-python

WORKDIR /workspace

# Copy Rust workspace root files and all non-Python lancedb source.
COPY lancedb/Cargo.toml lancedb/Cargo.lock lancedb/rust-toolchain.toml lancedb/
COPY lancedb/rust/ lancedb/rust/
# Cargo workspace lists nodejs as a member; it needs a Cargo.toml and stub
# source file to satisfy manifest parsing even though we never compile it.
COPY lancedb/nodejs/Cargo.toml lancedb/nodejs/
RUN mkdir -p lancedb/nodejs/src && touch lancedb/nodejs/src/lib.rs

# Copy the entire python directory (maturin requires the Python source present
# when building the wheel so it can bundle it alongside the compiled extension).
COPY lancedb/LICENSE lancedb/
COPY lancedb/python/ lancedb/python/

# Compile the Rust extension and install the resulting wheel.
#
# The --mount=type=cache lines persist the Cargo registry and compiled
# artifacts between docker build invocations on the same machine. When only
# Python source changes, the COPY layer above is invalidated and this RUN
# re-executes, but cargo finds all artifacts already compiled and maturin
# simply re-packages — taking seconds rather than minutes.
# CARGO_TARGET_DIR is set outside the workspace to avoid overlayfs interactions
# between the cache mount and the source tree.
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/cargo-target \
    export CARGO_TARGET_DIR=/cargo-target && \
    cd lancedb/python && \
    maturin build --release --out /tmp/wheels && \
    pip install /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy the compiled extension into the source tree so that the sys.path.insert
# in the training scripts can find it alongside the Python files.
RUN python -c "\
import shutil, glob, os, lancedb; \
so = glob.glob(os.path.join(os.path.dirname(lancedb.__file__), '_lancedb*.so'))[0]; \
shutil.copy(so, '/workspace/lancedb/python/python/lancedb/')"

# Copy training scripts.
COPY prepare_data.py train.py ./

# Default: run training. Override with `docker run ... python prepare_data.py`
# for data preparation, or with torchrun for multi-GPU:
#
#   Single node (4 GPUs):
#     docker run --gpus all -e LANCEDB_URI=... -e CHECKPOINT_URI=... \
#       torchrun --nproc_per_node=4 train.py
#
#   Two nodes (4 GPUs each) — run on both machines simultaneously:
#     docker run --gpus all --network host \
#       -e LANCEDB_URI=... -e CHECKPOINT_URI=... \
#       torchrun --nproc_per_node=4 --nnodes=2 \
#         --node_rank=<0 or 1> --master_addr=<node0_ip> --master_port=29500 \
#         train.py
CMD ["python", "train.py"]
