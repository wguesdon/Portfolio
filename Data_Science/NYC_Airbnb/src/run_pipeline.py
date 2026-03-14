"""
Run the full NYC Airbnb analysis pipeline.

    uv run python src/run_pipeline.py            # scripts only
    uv run python src/run_pipeline.py --report   # scripts + Quarto report
"""
import argparse
import subprocess
import sys
from pathlib import Path

STEPS = ["eda.py", "train.py"]


def run_scripts(src_dir: Path) -> None:
    for script in STEPS:
        sep = "=" * 60
        print(f"\n{sep}\n  {script}\n{sep}")
        result = subprocess.run(
            [sys.executable, str(src_dir / script)],
            cwd=src_dir,
        )
        if result.returncode != 0:
            print(f"\nERROR: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)


def render_report(project_root: Path) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  Rendering report.qmd\n{sep}")
    result = subprocess.run(
        ["quarto", "render", "report.qmd"],
        cwd=project_root,
    )
    if result.returncode != 0:
        print(f"\nERROR: quarto render failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  Report → {project_root / 'report.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NYC Airbnb analysis pipeline")
    parser.add_argument("--report", action="store_true",
                        help="Render the Quarto HTML report after running scripts")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip scripts; only render the Quarto report")
    args = parser.parse_args()

    src_dir = Path(__file__).parent
    project_root = src_dir.parent

    if not args.report_only:
        run_scripts(src_dir)

    if args.report or args.report_only:
        render_report(project_root)

    print("\n\nPipeline complete.")


if __name__ == "__main__":
    main()
