# How to Run Experiments on Hulk

## Machine Overview

| Resource | Spec |
|----------|------|
| Hostname | `hulk` |
| CPU | 96 cores, Intel Xeon Gold 6248R @ 3.00 GHz |
| RAM | 503 GB |
| GPUs | 4x NVIDIA A100-PCIE-40GB |
| Disk | 22 TB at `/DATATWO` (shared) |
| OS | Ubuntu, kernel 5.4.0-124 |

---

## 1. Python Environment

The shared venv lives at `/DATATWO/users/solubility/myenv/`. System Python is 3.10 but the venv has all the ML packages.

```bash
# Always use the venv python -- system python has nothing installed
PYTHON=/DATATWO/users/solubility/myenv/bin/python3

# Quick check
$PYTHON -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Install a new package
$PYTHON -m pip install <package>
```

There is no `.bashrc` alias set up, so you must either:
- Use the full path: `/DATATWO/users/solubility/myenv/bin/python3`
- Or add it to your PATH for the session: `export PATH="/DATATWO/users/solubility/myenv/bin:$PATH"`

---

## 2. GPU Management

### Check GPU status

```bash
# Quick overview: index, name, free memory, utilization
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader

# Full dashboard
nvidia-smi

# Live monitoring (updates every 2s)
watch -n 2 nvidia-smi
```

### Select a GPU

Use `CUDA_VISIBLE_DEVICES` to pin your process to a specific GPU. **Always check which GPUs are free first.**

```bash
# Run on GPU 2
CUDA_VISIBLE_DEVICES=2 $PYTHON my_script.py

# Run on GPUs 0 and 1 (multi-GPU)
CUDA_VISIBLE_DEVICES=0,1 $PYTHON my_script.py

# Force CPU only
CUDA_VISIBLE_DEVICES="" $PYTHON my_script.py
```

Inside Python, you can also select programmatically:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # must be BEFORE importing torch

import torch
device = torch.device("cuda:0")  # this is now physical GPU 2
```

**Important:** When you set `CUDA_VISIBLE_DEVICES=2`, torch only sees one GPU and calls it `cuda:0`. Don't use `cuda:2` after setting the env var.

### GPU etiquette

- Check `nvidia-smi` before claiming a GPU
- Don't hog all 4 GPUs for single-GPU workloads
- If you see someone else's process on a GPU, use a different one

---

## 3. CPU Management

The machine has 96 cores. Be mindful of shared usage.

### Limit CPU threads

Many libraries (numpy, torch, LightGBM) will grab all cores by default. Limit them:

```bash
# Environment variables (set BEFORE running your script)
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

$PYTHON my_script.py
```

Or in Python:

```python
import torch
torch.set_num_threads(8)
```

For LightGBM / CatBoost, use their `num_threads` / `thread_count` parameters:

```python
lgb.train(params, ..., params={"num_threads": 8})
CatBoostRegressor(thread_count=8, ...)
```

**Rule of thumb:** Don't use more than 30 cores for a single experiment.

---

## 4. Running Long Experiments

### Use tmux (recommended)

tmux sessions survive SSH disconnects. Always use tmux for anything that takes more than a few minutes.

```bash
# Create a named session
tmux new-session -d -s myexperiment "CUDA_VISIBLE_DEVICES=2 /DATATWO/users/solubility/myenv/bin/python3 -u my_script.py > output.log 2>&1"

# List sessions
tmux list-sessions

# Attach to see live output
tmux attach -t myexperiment

# Detach (without killing): Ctrl+B, then D

# Kill a session when done
tmux kill-session -t myexperiment
```

### Unbuffered output

Python buffers stdout when redirected to a file. Use `-u` for real-time log output:

```bash
# BAD: log file won't update until buffer flushes (could be minutes)
$PYTHON my_script.py > output.log 2>&1

# GOOD: log updates in real-time
stdbuf -oL $PYTHON -u my_script.py > output.log 2>&1
```

### Full tmux + GPU + logging pattern

```bash
PYTHON=/DATATWO/users/solubility/myenv/bin/python3

# Check free GPU
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader

# Launch on free GPU (e.g., GPU 3)
tmux new-session -d -s scaling \
  "cd /DATATWO/users/solubility/sc3-benchmark && \
   CUDA_VISIBLE_DEVICES=3 stdbuf -oL $PYTHON -u my_experiment.py > my_experiment.log 2>&1"

# Monitor progress
tail -f my_experiment.log

# Or check periodically
grep "PS-RMSE\|FRACTION\|Error" my_experiment.log
```

---

## 5. Running SC3 Benchmark Models

### Project structure

```
sc3-benchmark/
  data/splits/          # bench_train.csv, bench_eval.csv, bench_ood.csv
  data/sc3/             # sc3_hard.csv, sc3_medium.csv, sc3_easy.csv
  src/benchmarks/       # evaluate.py, data_splits.py, featurizers/, methods/
  scripts/              # run_cpu_baselines.py, run_gnn_baselines.py, etc.
  results/              # per-method results (summary.json, raw_results.json)
  Additional_Experiments/  # scaling, transfer, interpretability, etc.
```

### CPU models (trees, MLP)

```bash
cd /DATATWO/users/solubility/sc3-benchmark

# All CPU baselines (5 seeds, ~30 min)
OMP_NUM_THREADS=8 $PYTHON scripts/run_cpu_baselines.py

# Specific methods
$PYTHON scripts/run_cpu_baselines.py --methods lgb_rdkit catboost_rdkit --seeds 42 101 123

# Quick smoke test (1 seed)
$PYTHON scripts/run_cpu_baselines.py --quick

# Morgan fingerprint variants
$PYTHON scripts/run_morgan_baselines.py
```

### GNN models (GPU)

```bash
# All GNNs (GCN, GAT, GIN) on auto-selected GPU
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_gnn_baselines.py

# Single type, quick test
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/run_gnn_baselines.py --gnn_type GCN --quick
```

### Chemprop (D-MPNN)

Chemprop uses subprocess calls (`chemprop_train` / `chemprop_predict`). Make sure the venv bin is on PATH:

```bash
export PATH="/DATATWO/users/solubility/myenv/bin:$PATH"
CUDA_VISIBLE_DEVICES=2 $PYTHON -c "
from src.benchmarks.methods.chemprop_method import train_chemprop, predict_chemprop
# ... (see Additional_Experiments/scaling/run_scaling.py for full example)
"
```

**Note:** Chemprop v1.6.1 is installed. It had a numpy v2 compatibility issue that was patched in-place.

### Uni-Mol2 (foundation model)

```bash
CUDA_VISIBLE_DEVICES=2 $PYTHON -c "
from src.benchmarks.methods.unimol_method import extract_unimol_representations
reprs = extract_unimol_representations(smiles_list, use_cuda=True)
"
```

Extraction takes ~5 min for ~1300 molecules. Representations are cached per-session.

---

## 6. Running Additional Experiments

Each experiment lives in `Additional_Experiments/<name>/` and is self-contained.

```bash
cd /DATATWO/users/solubility/sc3-benchmark

# Scaling experiment
CUDA_VISIBLE_DEVICES=2 $PYTHON Additional_Experiments/scaling/run_scaling.py --gpu 2

# Check existing experiments
ls Additional_Experiments/
# heteroscedastic  interpretability  scaling  solvent_representation  transfer  transfer_gnn  transfer_v2
```

### Writing a new experiment

Keep it self-contained in `Additional_Experiments/<your_name>/`. Read from `../../data/` and `../../src/benchmarks/` but don't modify them. Your script should:

1. Import from `src.benchmarks` (add project root to `sys.path`)
2. Accept `--gpu` flag
3. Log to a file for tmux runs
4. Save results as JSON + a `findings.md`

```python
import sys
sys.path.insert(0, "/DATATWO/users/solubility/sc3-benchmark")
from src.benchmarks.data_splits import load_all_splits
from src.benchmarks.evaluate import compute_metrics
```

**Featurizer import caveat:** The `mordred` package is broken under numpy v2. Import featurizers directly to avoid the `__init__.py`:

```python
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

rdkit_feat = load_module("rdkit_featurizer",
    "/DATATWO/users/solubility/sc3-benchmark/src/benchmarks/featurizers/rdkit_featurizer.py")
RDKitFeaturizer = rdkit_feat.RDKitFeaturizer
```

---

## 7. Proxy

No HTTP/HTTPS proxy is configured on this machine. Outbound internet access is direct. If you encounter connectivity issues (e.g., downloading pretrained weights), check:

```bash
# Test connectivity
curl -I https://huggingface.co 2>&1 | head -3

# If a proxy is needed in the future, set:
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
export NO_PROXY=localhost,127.0.0.1
```

pip and git will respect these environment variables automatically.

---

## 8. Monitoring and Debugging

```bash
# Who's using the GPUs?
nvidia-smi

# Who's using CPU?
htop                    # interactive (if installed)
top -u solubility       # your processes only

# Check disk usage
du -sh /DATATWO/users/solubility/sc3-benchmark/

# Find and kill runaway processes
ps aux | grep python | grep -v grep
kill <PID>

# Check tmux sessions
tmux list-sessions

# Read a running process's output (if redirected to file)
tail -f output.log
```

---

## Quick Reference

```bash
# The one-liner you'll use most often:
PYTHON=/DATATWO/users/solubility/myenv/bin/python3
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader  # pick a free GPU
tmux new-session -d -s myrun \
  "cd /DATATWO/users/solubility/sc3-benchmark && \
   CUDA_VISIBLE_DEVICES=<GPU> stdbuf -oL $PYTHON -u my_script.py > run.log 2>&1"
tail -f run.log
```
