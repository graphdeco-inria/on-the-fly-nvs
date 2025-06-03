FROM nvidia/cuda:12.8.0-devel-ubuntu24.04
LABEL maintainer="Shady Gmira <shady.gmira@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

# --- Base packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-full \
    sudo \
    x11-apps \
    zenity \
    openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python requirements
COPY ./docker/requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir --break-system-packages torch torchvision xformers --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --break-system-packages cupy-cuda12x
# Modify this line to match your GPU Architecture
ENV TORCH_CUDA_ARCH_LIST="8.6" 
COPY ./submodules submodules
RUN pip install --no-cache-dir --break-system-packages --ignore-installed -r requirements.txt

# --- User Setup
ARG USER_UID
ARG USER_GID
ARG USERNAME

RUN set -eux; \
# Remove user with same UID if exists
EXISTING_USER_BY_UID=$(getent passwd "${USER_UID}" | cut -d: -f1 || true); \
if [ -n "$EXISTING_USER_BY_UID" ]; then \
userdel -r "$EXISTING_USER_BY_UID" || true; \
fi; \
\
# Remove user with same name if exists
if id "${USERNAME}" >/dev/null 2>&1; then \
userdel -r "${USERNAME}" || true; \
fi; \
\
# Remove group with same GID if exists
if getent group "${USER_GID}" >/dev/null; then \
groupdel "$(getent group "${USER_GID}" | cut -d: -f1)"; \
fi; \
\
# Remove group with same name if exists
if getent group "${USERNAME}" >/dev/null; then \
groupdel "${USERNAME}"; \
fi; \
\
# Now create group and user cleanly
groupadd --gid "${USER_GID}" "${USERNAME}"; \
useradd -m -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"; \
usermod -aG sudo "${USERNAME}"; \
echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# --- Include this last to avoid cache invalidation
WORKDIR /
# No clue why this is needed, but it is
# Otherwise, build of submodules/diff-gaussian-rasterization fails
RUN sed -i '1i#include <stdint.h>\n#include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h
RUN pip install --no-cache-dir --break-system-packages submodules/diff-gaussian-rasterization

# Set environment
ENV USER=${USERNAME}
ENV HOME=/home/${USERNAME}
WORKDIR /home/${USERNAME}
