"""Command line for one client file."""

import argparse
import json
from pathlib import Path

from jobkit.audit import load_table, prepare
from jobkit.classify import run_classify
from jobkit.forecast import run_forecast
from jobkit.report import write_report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a small-business forecast or scoring job from one table."
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument("table")
    parser.add_argument("--task", required=True, choices=["forecast", "classify"])
    parser.add_argument("--target", required=True, help="Column to predict")
    parser.add_argument("--date", help="Date column. Required for a forecast when it cannot be guessed.")
    parser.add_argument("--output", default="jobkit-output", help="Folder for the report, chart, and CSV")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_table(args.table)
    prepared, meta = prepare(frame, args.task, args.target, args.date)
    if args.task == "forecast":
        result = run_forecast(prepared, args.target, meta["date_column"], _Log(meta), output_dir)
    else:
        result = run_classify(
            prepared,
            args.target,
            meta["id_columns"],
            meta["feature_columns"],
            _Log(meta),
            output_dir,
        )
    result["leakage_suspects"] = meta["leakage_suspects"]
    result["id_columns"] = meta["id_columns"]
    report = write_report(result, output_dir)
    (output_dir / "audit.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Report: {report}")
    if result["task"] == "forecast":
        print(
            f"Holdout MAE baseline {result['baseline_mae']:.2f} vs selected {result['model_mae']:.2f}. "
            f"Better than baseline: {result['beats_baseline']}."
        )
    else:
        print(
            f"Holdout AUC selected {result['test_auc'][result['selected_model']]:.3f}. "
            f"Better than baseline: {result['beats_baseline']}."
        )


class _Log:
    """Append modeling notes onto the audit log that prepare() already started."""

    def __init__(self, meta):
        self.meta = meta

    def add(self, action, detail):
        self.meta["changes"].append({"action": action, "detail": detail})

    def as_dicts(self):
        return list(self.meta["changes"])
