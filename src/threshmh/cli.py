"""Command-line entry points. Driven by the Makefile."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def cmd_run(args) -> int:
    from .runner import run_experiment

    cfg = yaml.safe_load(Path(args.config).read_text())
    df, _ = run_experiment(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[cli] wrote {len(df)} rows to {out}")
    return 0


def cmd_scaling(args) -> int:
    from .scaling import run_scaling

    run_scaling(Path(args.out), max_classes=args.max_classes, skimage_max=args.skimage_max)
    return 0


def cmd_metrics(args) -> int:
    from .metrics_experiment import run_metrics_experiment

    run_metrics_experiment(Path(args.out))
    return 0


def cmd_report(args) -> int:
    from .report.build import build_all

    build_all(Path(args.results), Path(args.scaling), Path(args.metrics), Path(args.outdir))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="threshmh")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the main optimisation experiment")
    r.add_argument("--config", default="experiments/exp01/config.yaml")
    r.add_argument("--out", default="results/exp01_runs.csv")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("scaling", help="exact DP vs skimage timing")
    s.add_argument("--out", default="results/exp02_scaling.csv")
    s.add_argument("--max-classes", type=int, default=8)
    s.add_argument("--skimage-max", type=int, default=6)
    s.set_defaults(func=cmd_scaling)

    m = sub.add_parser("metrics", help="reconstruction vs segmentation metrics")
    m.add_argument("--out", default="results/exp03_metrics.csv")
    m.set_defaults(func=cmd_metrics)

    b = sub.add_parser("report", help="regenerate every table and figure")
    b.add_argument("--results", default="results/exp01_runs.csv")
    b.add_argument("--scaling", default="results/exp02_scaling.csv")
    b.add_argument("--metrics", default="results/exp03_metrics.csv")
    b.add_argument("--outdir", default="paper")
    b.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
