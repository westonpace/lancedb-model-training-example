# NGC PyTorch base includes CUDA, cuDNN, NCCL, and PyTorch pre-installed.
FROM nvcr.io/nvidia/pytorch:24.10-py3

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

# Copy the lancedb submodule source and build the Rust extension.
# maturin develop places the compiled _lancedb.so alongside the Python source,
# which is where the sys.path.insert in the training scripts expects to find it.
WORKDIR /workspace
COPY lancedb/ lancedb/
RUN cd lancedb/python && maturin develop --release

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
