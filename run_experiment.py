"""Entry point: repeat the experiment over multiple seeds and optimizers.

Usage:
    python run_experiment.py --optimizers gd --seeds 0 1 2
    python run_experiment.py --optimizers newton-cg lbfgs nelder-mead cma-es

The four study optimizers are imported from the ``optimizers/`` package;
until they are implemented they raise NotImplementedError, so the
default run uses the "gd" placeholder baseline to exercise the pipeline.
"""

from __future__ import annotations

import argparse

from src.data import load_and_split
from src.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizers", nargs="+", default=["gd"],
        help="optimizer names, e.g. gd newton-cg lbfgs nelder-mead cma-es",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                        help="random seeds for the data split")
    parser.add_argument("--init-seeds", nargs="+", type=int, default=None,
                        help="optional weight-init seeds (defaults to split seeds)")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    init_seeds = args.init_seeds if args.init_seeds is not None else args.seeds

    header = (
        f"{'optimizer':<12} {'seed':>4} {'final loss':>11} {'obj evals':>10} "
        f"{'grad evals':>11} {'time (s)':>9} {'test acc':>9} {'test F1':>8} "
        f"{'AUC':>7} {'mal recall':>11}"
    )
    print(header)
    print("-" * len(header))

    for seed, init_seed in zip(args.seeds, init_seeds):
        data = load_and_split(seed=seed)
        for name in args.optimizers:
            try:
                report = run_training(
                    data, optimizer_name=name, seed=seed, init_seed=init_seed,
                    max_iter=args.max_iter, tol=args.tol,
                )
            except NotImplementedError as err:
                print(f"{name:<12} {seed:>4}  NOT IMPLEMENTED -> {err}")
                continue
            r, t = report.result, report.test_metrics
            print(
                f"{name:<12} {seed:>4} {r.final_loss:>11.6f} "
                f"{r.n_objective_evals:>10} {r.n_gradient_evals:>11} "
                f"{r.wall_time:>9.3f} {t.accuracy:>9.4f} {t.f1:>8.4f} "
                f"{t.roc_auc:>7.4f} {t.recall_malignant:>11.4f}"
            )


if __name__ == "__main__":
    main()
