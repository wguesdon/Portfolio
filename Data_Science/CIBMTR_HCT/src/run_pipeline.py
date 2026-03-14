"""
CIBMTR — Full Pipeline

    uv run python src/run_pipeline.py            # EDA + train + ensemble
    uv run python src/run_pipeline.py --eda      # EDA only
    uv run python src/run_pipeline.py --train    # train + ensemble only
    uv run python src/run_pipeline.py --report   # pipeline + Quarto report
"""
import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent
ROOT = SRC.parent


def run(script: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {script}\n{sep}")
    result = subprocess.run(
        [sys.executable, str(SRC / script)],
        cwd=SRC,
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def render_report() -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  Rendering report.qmd\n{sep}")
    result = subprocess.run(
        ["quarto", "render", "report.qmd"],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"\nERROR: quarto render failed (code {result.returncode})")
        sys.exit(result.returncode)
    print(f"  Report → {ROOT / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CIBMTR HCT Analysis Pipeline")
    parser.add_argument("--eda",    action="store_true", help="EDA only")
    parser.add_argument("--train",  action="store_true", help="Train + ensemble only")
    parser.add_argument("--report", action="store_true", help="Also render Quarto report")
    args = parser.parse_args()

    run_eda   = args.eda   or (not args.eda and not args.train)
    run_train = args.train or (not args.eda and not args.train)

    if run_eda:
        run("eda.py")
    if run_train:
        run("train.py")
        run("ensemble.py")

    if args.report:
        render_report()

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
