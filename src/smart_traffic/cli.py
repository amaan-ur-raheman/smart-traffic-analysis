"""Command-line interface.

Everything the Streamlit app can do from the browser is also doable from a
terminal, which matters because ingestion of a 3 GB file should be a visible,
logged process rather than a progress bar.

Commands:
    sta build      Convert the raw CSV into Parquet and build the sample
    sta sample     Rebuild only the reproducibility sample
    sta train      Train the models and persist them
    sta validate   Print the data-quality report
    sta status     Show what is currently loaded
    sta demo       Generate a synthetic dataset for development
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from smart_traffic.config import PATHS, SETTINGS
from smart_traffic.logging_config import setup_logging


def _progress(stage: str, fraction: float, message: str) -> None:
    """Render ingestion progress as a single updating line."""
    bar = "#" * int(fraction * 30)
    sys.stdout.write(f"\r  [{bar:<30}] {fraction * 100:5.1f}%  {message[:60]:<60}")
    sys.stdout.flush()
    if stage == "done":
        sys.stdout.write("\n")


def cmd_build(args: argparse.Namespace) -> int:
    """Convert the raw CSV into the processed Parquet table."""
    from smart_traffic.ingestion import IngestionError, build_processed_dataset

    setup_logging(log_file=PATHS.logs / "sta.log")
    source = Path(args.source) if args.source else PATHS.raw_csv

    print(f"Source : {source}")
    print(f"Output : {PATHS.accidents_parquet}")
    print(f"Method : {args.method}\n")

    try:
        report = build_processed_dataset(
            source, method=args.method, build_sample_too=not args.no_sample, progress=_progress
        )
    except IngestionError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nThe dataset is not bundled with this project. To obtain it:\n"
            f"  1. Download 'US Accidents (2016-2023)' from {PATHS.raw_csv.parent}\n"
            "     https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents\n"
            f"  2. Place US_Accidents_March23.csv at {PATHS.raw_csv}\n"
            "  3. Run: uv run sta build\n\n"
            "For a quick demonstration without the 3 GB file, run: uv run sta demo",
            file=sys.stderr,
        )
        return 1

    print()
    print(report.to_dataframe().to_string(index=False))
    print(f"\nDone in {report.seconds:.1f}s. Start the app with: uv run streamlit run app.py")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Rebuild the reproducibility sample."""
    from smart_traffic.ingestion import build_sample

    setup_logging(log_file=PATHS.logs / "sta.log")
    info = build_sample(row_count=args.rows, seed=args.seed)
    print(info.to_dataframe().to_string(index=False))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the severity models and persist them."""
    from smart_traffic.db import DB
    from smart_traffic.modeling import TrainingConfig, save_models, train_models

    setup_logging(log_file=PATHS.logs / "sta.log")
    DB.reset()

    if not DB.is_available:
        print(
            "ERROR: No processed dataset found. Run 'uv run sta build' first.",
            file=sys.stderr,
        )
        return 1

    models = args.models.split(",") if args.models else list(TrainingConfig().selected_models)
    config = TrainingConfig(
        sample_rows=args.rows,
        selected_models=tuple(models),
    )

    print(
        f"Training {len(config.selected_models)} model(s) "
        f"on up to {config.sample_rows:,} rows..."
    )
    started = time.perf_counter()
    trained = train_models(config, db=DB, progress=lambda s, f, m: print(f"  {m}"))
    elapsed = time.perf_counter() - started

    directory = save_models(trained)
    print(f"\nTrained in {elapsed:.1f}s. Artefacts written to {directory}\n")
    for model in trained.values():
        metrics = model.metrics
        print(
            f"  {model.label:<20} macro F1={metrics['Macro_F1']:.4f}  "
            f"balanced acc={metrics['Balanced_Accuracy']:.4f}  "
            f"accuracy={metrics['Accuracy']:.4f}"
        )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Print the data-quality report."""
    from smart_traffic.db import DB
    from smart_traffic.validation import missing_value_profile, validate_processed_table

    setup_logging(log_file=PATHS.logs / "sta.log")
    DB.reset()
    if not DB.is_available:
        print("ERROR: No processed dataset found. Run 'uv run sta build' first.", file=sys.stderr)
        return 1

    report = validate_processed_table(DB)
    print(report.to_dataframe().to_string(index=False))
    print(
        f"\n{report.passed} passed, {report.warnings} warnings, {report.failures} failures "
        f"across {len(report.checks)} checks."
    )

    if args.missing:
        print("\nMissing values by column (top 20):")
        print(missing_value_profile(DB).head(20).to_string(index=False))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show what is currently available on disk."""
    from smart_traffic.db import DB
    from smart_traffic.ingestion import memory_footprint
    from smart_traffic.modeling import available_models

    setup_logging(log_file=PATHS.logs / "sta.log")
    print("Project root :", PATHS.root)
    print("Raw CSV      :", PATHS.raw_csv, "(present)" if PATHS.raw_csv.exists() else "(missing)")

    footprint = memory_footprint()
    if footprint["exists"]:
        print(
            f"Processed    : {footprint['path']} "
            f"({footprint['size_mb']} MB, {footprint['rows']:,} rows, "
            f"{footprint['columns']} columns)"
        )
        DB.reset()
        if DB.is_available:
            print("States       :", DB.scalar("SELECT COUNT(DISTINCT State) FROM accidents"))
            span = DB.sql("SELECT MIN(Start_Time) AS lo, MAX(Start_Time) AS hi FROM accidents")
            print("Coverage     :", span["lo"][0], "->", span["hi"][0])
    else:
        print("Processed    : (not built) - run 'uv run sta build'")

    sample = PATHS.sample_parquet
    print("Sample       :", sample, "(present)" if sample.exists() else "(not built)")

    models = available_models()
    print("Models       :", ", ".join(models) if models else "(none trained)")

    print(f"\nRush hour    : {SETTINGS.rush_hour_morning[0]:02d}:00-"
          f"{SETTINGS.rush_hour_morning[1]:02d}:59 and "
          f"{SETTINGS.rush_hour_evening[0]:02d}:00-{SETTINGS.rush_hour_evening[1]:02d}:59")
    print(f"Random seed  : {SETTINGS.random_seed}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate a synthetic dataset so the app can be explored immediately."""
    from smart_traffic.synthetic import SyntheticSpec, write_synthetic_csv

    setup_logging(log_file=PATHS.logs / "sta.log")
    PATHS.ensure_dirs()
    target = Path(args.output) if args.output else (PATHS.synthetic / "synthetic_accidents.csv")

    spec = SyntheticSpec(rows=args.rows, seed=args.seed)
    print("Generating a SYNTHETIC dataset for development.")
    print("These records are not real accidents and must never be reported as results.\n")
    write_synthetic_csv(target, spec)
    print(f"Written: {target} ({target.stat().st_size / 1e6:.1f} MB, {args.rows:,} rows)")
    print(f"\nNext: uv run sta build --source {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="sta",
        description="Smart Traffic Analysis and Accident Severity Prediction",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Convert the raw CSV into processed Parquet")
    build.add_argument("--source", help=f"Path to the CSV (default: {PATHS.raw_csv})")
    build.add_argument(
        "--method", choices=["chunked", "duckdb"], default="chunked",
        help="chunked: low memory, exact progress. duckdb: faster, coarser progress.",
    )
    build.add_argument("--no-sample", action="store_true", help="Skip building the ML sample")
    build.set_defaults(func=cmd_build)

    sample = subparsers.add_parser("sample", help="Rebuild the reproducibility sample")
    sample.add_argument("--rows", type=int, default=SETTINGS.sample_training_rows)
    sample.add_argument("--seed", type=int, default=SETTINGS.random_seed)
    sample.set_defaults(func=cmd_sample)

    train = subparsers.add_parser("train", help="Train and persist the severity models")
    train.add_argument("--rows", type=int, default=SETTINGS.sample_training_rows)
    train.add_argument(
        "--models", help="Comma-separated model keys (default: all three)"
    )
    train.set_defaults(func=cmd_train)

    validate = subparsers.add_parser("validate", help="Print the data-quality report")
    validate.add_argument("--missing", action="store_true",
        help="Include the missing-value profile")
    validate.set_defaults(func=cmd_validate)

    status = subparsers.add_parser("status", help="Show what is currently available")
    status.set_defaults(func=cmd_status)

    demo = subparsers.add_parser("demo", help="Generate a synthetic development dataset")
    demo.add_argument("--rows", type=int, default=60_000)
    demo.add_argument("--seed", type=int, default=20240101)
    demo.add_argument("--output", help="Destination CSV path")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument vector. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        # The traceback belongs in the log; the terminal gets an actionable line.
        setup_logging(log_file=PATHS.logs / "sta.log").exception("Command failed")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
