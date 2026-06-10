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

# Copy only the files required for Rust compilation. Keeping these in their own
# layers means that changes to the pure Python source below do not invalidate
# the Rust build cache.
WORKDIR /workspace
COPY lancedb/Cargo.toml lancedb/Cargo.lock lancedb/rust-toolchain.toml lancedb/
COPY lancedb/rust/ lancedb/rust/
COPY lancedb/nodejs/Cargo.toml lancedb/nodejs/
# Cargo validates that src/lib.rs exists during manifest parsing. Create a
# stub so cargo metadata succeeds without needing the full nodejs source.
RUN mkdir -p lancedb/nodejs/src && touch lancedb/nodejs/src/lib.rs
COPY lancedb/python/Cargo.toml lancedb/python/build.rs lancedb/python/pyproject.toml lancedb/python/README.md lancedb/python/
COPY lancedb/python/src/ lancedb/python/src/

# Compile the Rust extension (~5-10 min, cached until Rust source changes).
# maturin develop requires a virtualenv which the NGC container doesn't have,
# so we build a wheel and pip-install it into the system Python instead.
RUN cd lancedb/python && \
    maturin build --release --out /tmp/wheels && \
    pip install /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy pure Python source in a separate layer. Changes here skip the Rust build.
COPY lancedb/python/python/ lancedb/python/python/

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
