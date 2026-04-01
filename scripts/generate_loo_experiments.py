"""Enumerate LOO experiments and write one CLI invocation per line.

Usage:
    python scripts/generate_loo_experiments.py \
        --data_mode balanced --run_name hyperpar

Output: scripts/loo_experiments.txt  (one line per experiment)
Then:   sbatch --array=0-$(( $(wc -l < scripts/loo_experiments.txt) - 1 )) scripts/loo.sh
"""

import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loo import model_path_iter  # noqa: E402


def main():
    parser = ArgumentParser()
    parser.add_argument("--data_mode", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/mr2238/scratch_pi_np442/mr2238/accelerate",
    )
    parser.add_argument(
        "--patient_balance", type=str, default="weight",
    )
    parser.add_argument(
        "--output", type=str, default="scripts/loo_experiments.txt",
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.base_dir, args.data_mode)
    dataset_names = sorted(
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and "chop" in d
    )

    lines = []
    seen = set()
    for exp_path, is_debug, dataset, model_name in model_path_iter(
        train_dir, dataset_names, args.run_name
    ):
        # Check that a pkl exists for this model
        pkl_path = os.path.join(os.path.dirname(exp_path), f"{model_name}.pkl")
        if not os.path.exists(pkl_path):
            continue

        key = (dataset, model_name)
        if key in seen:
            continue
        seen.add(key)

        line = (
            f"--data_mode {args.data_mode} "
            f"--run_name {args.run_name} "
            f"--dataset {dataset} "
            f"--model_filter {model_name} "
            f"--patient_balance {args.patient_balance}"
        )
        lines.append(line)

    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} experiments to {args.output}")


if __name__ == "__main__":
    main()
