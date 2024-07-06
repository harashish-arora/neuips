"""
CLI entry point: python -m explainer --api-keys KEY1 KEY2 ...
"""

import argparse
from .config import PipelineConfig
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Glass-Onion explanation pipeline"
    )
    parser.add_argument(
        "--api-keys", nargs="+", required=True,
        help="One or more Gemini API keys",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="enhanced_explanations")
    parser.add_argument(
        "--feature-thresholds", type=str,
        default="threshold_analysis/feature_thresholds.json",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        feature_thresholds_file=args.feature_thresholds,
    )
    run(args.api_keys, cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
