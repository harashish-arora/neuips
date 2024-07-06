#!/usr/bin/env bash
# Workaround for the broken /dev/nvidia0 on this host.
#
# /dev/nvidia0 is in an EIO state (the underlying GPU at PCI 0000:18:00.0 has
# fallen off the bus); PyTorch's CUDA driver always opens it during init,
# fails, and then refuses to use any GPU at all.  We work around this by
# entering an unprivileged user+mount namespace (unshare -r -m) and
# bind-mounting one of the *healthy* device files (1, 2, or 3) over
# /dev/nvidia0.  PyTorch then opens what it thinks is /dev/nvidia0 and gets
# a working GPU, while CUDA_VISIBLE_DEVICES=0 inside the namespace points to
# that bind-mounted device.
#
# Usage:
#   ./run_with_gpu.sh <healthy-gpu-index 1|2|3> <command...>
# Example:
#   ./run_with_gpu.sh 1 python run_transfer_298k.py --approach filter

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <healthy-gpu-index 1|2|3> <command...>" >&2
    exit 2
fi

GPU_IDX="$1"
shift

if [[ "$GPU_IDX" != "1" && "$GPU_IDX" != "2" && "$GPU_IDX" != "3" ]]; then
    echo "ERROR: GPU index must be 1, 2 or 3 (0 is broken on this host)" >&2
    exit 2
fi

exec unshare -r -m -- bash -c "
    mount --bind /dev/nvidia${GPU_IDX} /dev/nvidia0
    export CUDA_VISIBLE_DEVICES=0
    exec \"\$@\"
" -- "$@"
