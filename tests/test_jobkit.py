import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jobkit.cli import main


def _sales_csv(path):
    rng = np.random.default_rng(0)
    months = pd.date_range("2015-01-01", periods=96, freq="MS")
    sales = 200 + 1.5 * np.arange(96) + 40 * np.sin(2 * np.pi * months.month / 12)
    sales = sales + rng.normal(0, 2, 96)
    frame = pd.DataFrame({"month": months, "sales": sales, "store_id": "shop-1"})
    frame.loc[3, "sales"] = np.nan
    extra = frame.iloc[[10]].copy()
    frame = pd.concat([frame, extra], ignore_index=True)
    frame.to_csv(path, index=False)


def _leads_csv(path):
    rng = np.random.default_rng(1)
    rows = 500
    segment = rng.choice(["hot", "warm", "cold"], rows, p=[0.15, 0.35, 0.5])
    age = rng.normal(40, 8, rows)
    logit = np.where(segment == "hot", 1.4, np.where(segment == "warm", -0.2, -1.4))
    logit = logit + (age - 40) / 40
    probability = 1 / (1 + np.exp(-logit))
    bought = rng.binomial(1, probability)
    duration = np.where(bought == 1, rng.integers(180, 400, rows), rng.integers(5, 40, rows))
    pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "segment": segment,
            "age": age,
            "bought": np.where(bought == 1, "yes", "no"),
            "call_duration": duration,
        }
    ).to_csv(path, index=False)


class JobKitTests(unittest.TestCase):
    def test_forecast_writes_report_chart_and_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "sales.csv"
            output = root / "out"
            _sales_csv(source)
            main(
                [
                    "run",
                    str(source),
                    "--task",
                    "forecast",
                    "--target",
                    "sales",
                    "--date",
                    "month",
                    "--output",
                    str(output),
                ]
            )
            self.assertTrue((output / "REPORT.md").exists())
            self.assertTrue((output / "REPORT.html").exists())
            self.assertTrue((output / "forecast.png").exists())
            self.assertTrue((output / "predictions.csv").exists())
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("REVIEW:", report)
            self.assertIn("baseline", report.lower())
            predictions = pd.read_csv(output / "predictions.csv")
            self.assertGreater(len(predictions), 4)

    def test_classify_holds_out_leakage_and_writes_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "leads.csv"
            output = root / "out"
            _leads_csv(source)
            main(
                [
                    "run",
                    str(source),
                    "--task",
                    "classify",
                    "--target",
                    "bought",
                    "--output",
                    str(output),
                ]
            )
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            audit = (output / "audit.json").read_text(encoding="utf-8")
            self.assertIn("REVIEW:", report)
            self.assertIn("call_duration", audit)
            self.assertTrue((output / "scores.png").exists())
            scored = pd.read_csv(output / "scored_rows.csv")
            self.assertIn("score", scored.columns)
            self.assertGreater(scored["score"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
