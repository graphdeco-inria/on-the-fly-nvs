#!/bin/bash
set -e

# ── SSH setup ─────────────────────────────────────────────────────────────────
# Inject public key from environment variable if provided.
# Alternatively, mount your ~/.ssh/authorized_keys to /root/.ssh/authorized_keys.
if [ -n "${SSH_PUBLIC_KEY}" ]; then
    mkdir -p /root/.ssh
    echo "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

# Generate host keys if not already present (first start)
ssh-keygen -A

# Start SSH daemon in background
/usr/sbin/sshd

echo "SSH server started on port 22"

# ── Main process ──────────────────────────────────────────────────────────────
exec "$@"
