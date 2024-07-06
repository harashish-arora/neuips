#!/usr/bin/env python3
"""
Parallel benchmark runner — uses all available CPUs and GPUs.

CPU methods run in parallel processes, each with multiple threads.
GPU methods each get their own GPU.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def get_completed_methods():
    """Check which methods already have complete results (5 seeds)."""
    completed = set()
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        models_dir = d / "models"
        if models_dir.exists():
            n_seeds = len(list(models_dir.glob("*.pkl")))
            if n_seeds >= 5:
                completed.add(d.name)
        elif (d / "summary.json").exists():
            with open(d / "summary.json") as f:
                s = json.load(f)
            if s.get("n_seeds", 0) >= 5:
                completed.add(d.name)
    return completed


def run_method(method_name, threads=8, gpu_id=None):
    """Run a single method with proper resource allocation."""
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""

    cmd = [
        sys.executable, str(ROOT / "sc3"),
        "run", "--methods", method_name,
        "--threads", str(threads),
    ]

    print(f"[START] {method_name} (threads={threads}, gpu={gpu_id})")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=7200,
        )
        elapsed = time.time() - t0
        success = result.returncode == 0

        log_dir = RESULTS_DIR / method_name
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "run_log.txt", "w") as f:
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")

        status = "OK" if success else "FAIL"
        print(f"[{status}] {method_name} — {elapsed:.0f}s")
        if not success:
            last_lines = result.stderr.strip().split("\n")[-5:]
            print(f"       Error: {' '.join(last_lines)}")

        return method_name, success, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"[TIMEOUT] {method_name} — {elapsed:.0f}s")
        return method_name, False, elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-workers", type=int, default=4,
                        help="Max parallel CPU method processes")
    parser.add_argument("--threads-per-method", type=int, default=16,
                        help="CPU threads per method")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    args = parser.parse_args()

    completed = set() if args.force else get_completed_methods()
    print(f"Already completed: {completed or 'none'}")

    cpu_methods = [
        "rf_rdkit", "xgb_rdkit", "lgb_rdkit", "catboost_rdkit",
        "mlp_rdkit",
        "rf_morgan", "xgb_morgan", "lgb_morgan", "catboost_morgan",
        "fastprop", "dissolvr",
    ]

    gpu_methods = [
        ("gnn_gcn", 0),
        ("gnn_gat", 1),
        ("gnn_gin", 2),
        ("soltrannet", 3),
    ]

    cpu_todo = [m for m in cpu_methods if m not in completed]
    gpu_todo = [(m, g) for m, g in gpu_methods if m not in completed]

    print(f"\nCPU methods to run: {cpu_todo}")
    print(f"GPU methods to run: {[m for m, _ in gpu_todo]}")
    print(f"CPU workers: {args.cpu_workers}, threads/method: {args.threads_per_method}")
    print()

    all_results = []

    # Run CPU methods in parallel
    if cpu_todo:
        print(f"{'='*60}")
        print(f"Running {len(cpu_todo)} CPU methods ({args.cpu_workers} parallel)")
        print(f"{'='*60}")

        with ProcessPoolExecutor(max_workers=args.cpu_workers) as executor:
            futures = {
                executor.submit(run_method, m, args.threads_per_method, None): m
                for m in cpu_todo
            }
            for future in as_completed(futures):
                name, success, elapsed = future.result()
                all_results.append({"method": name, "success": success, "time": elapsed})

    # Run GPU methods in parallel (each on its own GPU)
    if gpu_todo:
        print(f"\n{'='*60}")
        print(f"Running {len(gpu_todo)} GPU methods (one per GPU)")
        print(f"{'='*60}")

        with ProcessPoolExecutor(max_workers=len(gpu_todo)) as executor:
            futures = {
                executor.submit(run_method, m, 4, g): m
                for m, g in gpu_todo
            }
            for future in as_completed(futures):
                name, success, elapsed = future.result()
                all_results.append({"method": name, "success": success, "time": elapsed})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in sorted(all_results, key=lambda x: x["time"]):
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['method']:<20s} {r['time']:>7.0f}s")

    n_ok = sum(1 for r in all_results if r["success"])
    print(f"\n{n_ok}/{len(all_results)} methods completed successfully.")


if __name__ == "__main__":
    main()
