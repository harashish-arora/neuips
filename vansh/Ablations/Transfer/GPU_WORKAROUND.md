# Working around the broken `/dev/nvidia0` on `hulk`

## The problem

`hulk` was originally a 4×A100 box (PCI bus IDs 18, 3B, 86, AF).  At
**2024-06-27 21:44:19 UTC** the GPU at PCI `0000:18:00.0` (which the
kernel exports as `/dev/nvidia0`) entered a hardware-failed state and
has been emitting `RmInitAdapter failed (0x22:0x56:714)` to the kernel
log on every probe ever since.

The other three GPUs (PCIs 3B / 86 / AF, exported as
`/dev/nvidia1`, `/dev/nvidia2`, `/dev/nvidia3`) are physically healthy
and `nvidia-smi -L` lists them.  But **PyTorch's CUDA driver opens
`/dev/nvidia0` first during initialisation regardless of
`CUDA_VISIBLE_DEVICES`** — that probe returns `EIO`, the runtime sets
the global error to `cudaErrorInvalidDevice` (101), and *every*
subsequent CUDA call fails until the process exits.  Net effect: even
though three GPUs are idle, PyTorch reports `torch.cuda.is_available()
== False` and the whole machine looks dead.

We confirmed this with `strace`:

```
$ CUDA_VISIBLE_DEVICES=GPU-...uuid... \
    strace -e trace=openat python -c 'import torch; torch.cuda.is_available()' \
    2>&1 | grep '/dev/nvidia[0-9]'
openat(AT_FDCWD, "/dev/nvidia0", O_RDWR|O_CLOEXEC) = -1 EIO (Input/output error)
openat(AT_FDCWD, "/dev/nvidia0", O_RDWR)            = -1 EIO (Input/output error)
```

i.e. PyTorch never even tries to open `/dev/nvidia1`, `/dev/nvidia2`,
or `/dev/nvidia3` — the first `EIO` poisons the runtime.

A reboot fixes it for everyone (the dead GPU 0 will probably need
physical replacement), but until then we need a non-root userspace
workaround.

## The workaround: user-namespace bind mount

Linux `unshare(2)` lets an unprivileged user create a private
**mount namespace** (with `-r -m`, you simultaneously enter a user
namespace where you appear to be `root`, just for the purposes of
mount inside that namespace — the rest of the system is unaffected).
Inside that private mount namespace we can:

1. bind-mount one of the *healthy* device files
   (`/dev/nvidia1`, `/dev/nvidia2`, or `/dev/nvidia3`) on top of
   `/dev/nvidia0`,
2. then run PyTorch with `CUDA_VISIBLE_DEVICES=0`.

PyTorch opens what it thinks is `/dev/nvidia0`, which the kernel
resolves through the bind mount to a healthy GPU device file, and the
runtime initialises happily.  The mount is **per-process** (and its
children) — no other user sees it, and it is automatically torn down
when the process exits.  No root, no kernel changes, no driver
restart.

The wrapper script in this directory (`run_with_gpu.sh`) implements
exactly that:

```bash
#!/usr/bin/env bash
set -euo pipefail
GPU_IDX="$1"     # one of 1, 2, 3 (the *physical* device index)
shift
exec unshare -r -m -- bash -c '
    mount --bind /dev/nvidia'"$GPU_IDX"' /dev/nvidia0
    export CUDA_VISIBLE_DEVICES=0
    exec "$@"
' -- "$@"
```

Usage:

```bash
# Run anything with GPU "1" (= physical /dev/nvidia1 = PCI 0000:3B:00.0)
./run_with_gpu.sh 1 python my_script.py
```

To run **multiple** GPUs in parallel for separate jobs (the whole
point of this trick) you launch each in its own tmux pane:

```bash
tmux new-session -d -s job1 './run_with_gpu.sh 1 python script.py'
tmux new-session -d -s job2 './run_with_gpu.sh 2 python script.py'
tmux new-session -d -s job3 './run_with_gpu.sh 3 python script.py'
```

Each tmux session has its own mount namespace; the three
`mount --bind` operations don't interfere with each other (they happen
in three private namespaces) and *also* don't affect anyone else on
the system.  `nvidia-smi` still shows all three healthy GPUs as in
use, just listed under a single "device 0" within each namespace.

## Verify

```bash
# Outside the workaround — broken:
$ python -c 'import torch; print(torch.cuda.is_available())'
False

# With the workaround — works:
$ ./run_with_gpu.sh 1 python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
True NVIDIA A100-PCIE-40GB
```

## Caveats

- This is a **workaround**.  It does not fix the dead GPU; it only
  hides the broken device file from one process tree at a time.  Any
  other user / process on `hulk` still hits the same problem unless
  they wrap their command the same way.
- The bind mount only lasts as long as the wrapper's child process.
  Killing the tmux session tears it down automatically.
- We bind-mount the **device file**, not the device.  All bookkeeping
  (UUIDs, SMI process accounting, etc.) still attributes the work to
  the real underlying GPU — the workaround is purely a userspace
  redirect of which file PyTorch opens.
- If the kernel device file count ever changes (e.g. someone replaces
  the dead GPU in a hot-add), `run_with_gpu.sh` would just keep doing
  the right thing as long as `/dev/nvidia1..3` continue to map to
  healthy GPUs.

## When to remove this

After `hulk` is rebooted (which will wipe out the bad CUDA driver
state and re-enumerate the device files cleanly), this workaround is
no longer needed and the wrapper can be removed — at that point the
normal `python ... --gpu N` arguments work as documented in
`vansh/README.md`.
