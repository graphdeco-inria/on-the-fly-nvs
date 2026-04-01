# On-the-fly NVS - Docker Image
# Requires NVIDIA GPU with CUDA 12.x driver support (nvidia-smi should show CUDA 12.x)
# Build: docker build -t on-the-fly-nvs .
# Submodules must be initialized before building:
#   git submodule update --init --recursive

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    XFORMERS_FORCE_DISABLE_TRITON=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    # Aliyun PyPI mirror for all pip installs (overridden per-command for PyTorch)
    PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    PIP_TRUSTED_HOST=mirrors.aliyun.com

# ── Ubuntu apt → Aliyun mirror ─────────────────────────────────────────────────
# Replaces only sources.list (Ubuntu packages); CUDA apt sources in
# sources.list.d/ are left untouched.
# Use -E and https? to handle both http:// and https:// variants.
RUN sed -i -E \
    -e 's|https?://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    -e 's|https?://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
    /etc/apt/sources.list

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    # OpenCV runtime deps
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    # glfw (graphdecoviewer) runtime — libglfw3.so is loaded via ctypes at import
    libx11-6 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default; do NOT set pip alternative here —
# ensurepip bootstraps pip3.12 into /usr/local/bin which takes PATH priority.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# ── PyTorch + CUDA 12.8 ────────────────────────────────────────────────────────
# PyTorch whl must come from the official index (no Aliyun mirror available),
# so --index-url overrides PIP_INDEX_URL for this step only.
RUN pip install --no-cache-dir \
    torch torchvision xformers \
    --index-url https://download.pytorch.org/whl/cu128

# cupy is on PyPI → uses Aliyun mirror via PIP_INDEX_URL
RUN pip install --no-cache-dir cupy-cuda12x

# ── Python dependencies (non-submodule) ───────────────────────────────────────
RUN pip install --no-cache-dir \
    plyfile \
    tqdm \
    opencv-python \
    lpips \
    websockets

# ── Copy source (submodules must be initialized locally first) ─────────────────
COPY . .

# ── Build custom CUDA extensions ───────────────────────────────────────────────
# --no-build-isolation: setup.py imports torch at the top level, so it must
# run in the same environment where torch is already installed, not in pip's
# isolated build env.
#
# TORCH_CUDA_ARCH_LIST: build-time container has no GPU, so torch cannot
# auto-detect architectures and raises "IndexError: list index out of range".
# Explicit list covers Turing(7.5) / Ampere(8.0,8.6) / Ada(8.9) / Hopper(9.0).
# The compiled .so runs on the actual GPU at container start time.
ENV MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir --no-build-isolation submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir --no-build-isolation submodules/fused-ssim
RUN pip install --no-cache-dir --no-build-isolation submodules/simple-knn
RUN pip install --no-cache-dir submodules/graphdecoviewer

# ── Cache volume mount points ──────────────────────────────────────────────────
RUN mkdir -p /cache/huggingface /cache/torch /app/results /app/data

EXPOSE 6009 8000

ENV STREAM_URL=""

CMD ["sh", "-c", \
    "python train.py \
        -s ${STREAM_URL} \
        --downsampling 1.5 \
        --viewer_mode web \
        -m /app/results/$(date +%Y%m%d_%H%M%S)"]
