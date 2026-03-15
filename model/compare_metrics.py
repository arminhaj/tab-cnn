"""Compare evaluation metrics between two saved TabCNN runs.

Usage:
  ./.venv/bin/python model/compare_metrics.py \
    --run-a "model/saved/c_cnn 2026-02-24 00-19-51" \
    --run-b "model/saved/c_crnn 2026-02-23 14-07-45" \
    --label-a cnn \
    --label-b crnn
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Metrics import (
    incorrect_note_distance,
    pitch_f_measure,
    pitch_precision,
    pitch_recall,
    tab_disamb,
    tab_f_measure,
    tab_precision,
    tab_recall,
)


METRIC_FNS = {
    "pp": pitch_precision,
    "pr": pitch_recall,
    "pf": pitch_f_measure,
    "tp": tab_precision,
    "tr": tab_recall,
    "tf": tab_f_measure,
    "tdr": tab_disamb,
    "ind": incorrect_note_distance,
}

# Most metrics: higher is better. `ind` is a distance where lower is better.
HIGHER_IS_BETTER = {"pp", "pr", "pf", "tp", "tr", "tf", "tdr"}


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _load_run_metrics(run_dir: Path) -> pd.DataFrame:
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
    if not fold_dirs:
        raise FileNotFoundError(f"No numeric fold directories found in: {run_dir}")

    rows = []
    labels = []
    for fold_dir in fold_dirs:
        pred_candidates = [fold_dir / "predictions.npz", fold_dir / "artifacts" / "predictions.npz"]
        pred_file = next((path for path in pred_candidates if path.exists()), None)
        if pred_file is None:
            raise FileNotFoundError(
                f"Missing predictions file in fold {fold_dir}. Checked: "
                + ", ".join(str(path) for path in pred_candidates)
            )
        with np.load(pred_file, allow_pickle=False) as loaded:
            y_pred = loaded["y_pred"]
            y_gt = loaded["y_gt"]
        rows.append({metric: fn(y_pred, y_gt) for metric, fn in METRIC_FNS.items()})
        labels.append(f"g{fold_dir.name}")

    return pd.DataFrame(rows, index=labels)


def _with_stats(df: pd.DataFrame) -> pd.DataFrame:
    mean_row = pd.DataFrame([df.mean(axis=0)], index=["mean"])
    std_row = pd.DataFrame([df.std(axis=0, ddof=0)], index=["std dev"])
    return pd.concat([df, mean_row, std_row])


def compare_runs(run_a: Path, run_b: Path, label_a: str, label_b: str, out_dir: Path) -> dict[str, Path]:
    run_a_df = _load_run_metrics(run_a)
    run_b_df = _load_run_metrics(run_b)

    run_a_all = _with_stats(run_a_df)
    run_b_all = _with_stats(run_b_df)

    common_index = run_a_all.index.intersection(run_b_all.index)
    run_a_all = run_a_all.loc[common_index]
    run_b_all = run_b_all.loc[common_index]

    per_fold = pd.concat({label_a: run_a_all, label_b: run_b_all}, axis=1)

    summary_rows = []
    for metric in METRIC_FNS:
        a_mean = float(run_a_all.loc["mean", metric])
        b_mean = float(run_b_all.loc["mean", metric])
        a_std = float(run_a_all.loc["std dev", metric])
        b_std = float(run_b_all.loc["std dev", metric])
        delta = b_mean - a_mean
        pct_change_vs_ref = np.nan if a_mean == 0 else (delta / a_mean) * 100.0

        higher_is_better = metric in HIGHER_IS_BETTER
        if higher_is_better:
            improvement_pct = pct_change_vs_ref
        else:
            # Lower is better; improvement is positive when candidate is lower.
            improvement_pct = np.nan if a_mean == 0 else ((a_mean - b_mean) / a_mean) * 100.0

        if higher_is_better:
            better = label_b if b_mean > a_mean else label_a if a_mean > b_mean else "tie"
        else:
            better = label_b if b_mean < a_mean else label_a if a_mean < b_mean else "tie"

        if better == "tie":
            verdict = "no change"
        elif better == label_b:
            verdict = f"{label_b} better"
        else:
            verdict = f"{label_b} worse"

        summary_rows.append(
            {
                "metric": metric,
                f"{label_a}_mean": a_mean,
                f"{label_a}_std": a_std,
                f"{label_b}_mean": b_mean,
                f"{label_b}_std": b_std,
                f"delta_{label_b}_minus_{label_a}": delta,
                f"pct_change_vs_{label_a}": pct_change_vs_ref,
                "improvement_pct": improvement_pct,
                "better": better,
                "verdict": verdict,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_label(label_a)}_vs_{_safe_label(label_b)}"
    summary_csv = out_dir / f"{prefix}_summary.csv"
    summary_md = out_dir / f"{prefix}_summary.md"
    per_fold_csv = out_dir / f"{prefix}_per_fold.csv"

    summary_df.to_csv(summary_csv, index=False)
    _write_markdown_table(summary_df, summary_md)
    per_fold.to_csv(per_fold_csv)

    return {"summary_csv": summary_csv, "summary_md": summary_md, "per_fold_csv": per_fold_csv}


def _format_md_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        row_values = [_format_md_value(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(row_values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics between two TabCNN saved runs.")
    parser.add_argument("--run-a", type=Path, required=True, help="Path to first run directory.")
    parser.add_argument("--run-b", type=Path, required=True, help="Path to second run directory.")
    parser.add_argument("--label-a", type=str, default="run_a", help="Label for first run.")
    parser.add_argument("--label-b", type=str, default="run_b", help="Label for second run.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("model/saved/comparisons"),
        help="Output directory for comparison files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = compare_runs(args.run_a, args.run_b, args.label_a, args.label_b, args.out_dir)
    print("Wrote comparison files:")
    for _, path in outputs.items():
        print(path)
