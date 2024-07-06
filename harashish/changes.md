# Changes to Speed Up Uni-Mol2

## `harashish/src/benchmarks/methods/unimol_method.py`
1. Increased robust processing parameters for the A100 GPU:
   - Modified `extract_unimol_representations` to take `batch_size=2048` (up from 32) and `chunk_size` of `10000` (up from 256).
2. Parallelized string canonicalization/sanitization steps over 8 CPU threads instead of sequentially running on 1 thread using `multiprocessing.Pool`.

## Custom Execution SDK (`harashish/scripts/run_unimol_catboost.py`)
1. Re-implemented the CLI explicitly into a reproducible Python script.
2. Forces the benchmark to load all correct `data_splits.py` data natively inside `harashish`.
3. Directly writes out outputs (models, seed metrics, and summary.json) securely into `/harashish/results/unimol_catboost/`.
