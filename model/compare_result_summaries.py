from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["pp", "pr", "pf", "tp", "tr", "tf", "tdr", "ind"]
HIGHER_IS_BETTER = {"pp", "pr", "pf", "tp", "tr", "tf", "tdr"}
OBJECTIVE_KEYS = [
    "closed_class_weight",
    "use_sounding_aux_loss",
    "sounding_aux_weight",
    "sounding_positive_weight",
    "use_class_weighting",
    "class_weight_power",
    "min_class_weight",
    "max_class_weight",
    "optimizer_learning_rate",
]


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "run"


def _read_results(run_dir: Path) -> tuple[pd.Series, pd.Series]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv: {results_path}")
    df = pd.read_csv(results_path)
    mean_row = df[df["data"] == "mean"]
    std_row = df[df["data"] == "std dev"]
    if mean_row.empty or std_row.empty:
        raise ValueError(f"results.csv in {run_dir} is missing mean/std dev rows")
    return mean_row.iloc[0], std_row.iloc[0]


def _read_objective_settings(run_dir: Path) -> dict[str, str]:
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log.txt: {log_path}")
    settings: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in OBJECTIVE_KEYS:
            settings[key] = value.strip()
    return settings


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_runs(run_a: Path, run_b: Path, label_a: str, label_b: str, out_dir: Path) -> dict[str, Path]:
    mean_a, std_a = _read_results(run_a)
    mean_b, std_b = _read_results(run_b)
    objective_a = _read_objective_settings(run_a)
    objective_b = _read_objective_settings(run_b)

    summary_rows = []
    for metric in METRICS:
        a_mean = float(mean_a[metric])
        b_mean = float(mean_b[metric])
        a_std = float(std_a[metric])
        b_std = float(std_b[metric])
        delta = b_mean - a_mean
        pct_change_vs_a = np.nan if a_mean == 0 else (delta / a_mean) * 100.0
        if metric in HIGHER_IS_BETTER:
            improvement_pct = pct_change_vs_a
            better = label_b if b_mean > a_mean else label_a if a_mean > b_mean else "tie"
        else:
            improvement_pct = np.nan if a_mean == 0 else ((a_mean - b_mean) / a_mean) * 100.0
            better = label_b if b_mean < a_mean else label_a if a_mean < b_mean else "tie"
        verdict = "no change" if better == "tie" else f"{label_b} better" if better == label_b else f"{label_b} worse"
        summary_rows.append(
            {
                "metric": metric,
                f"{label_a}_mean": a_mean,
                f"{label_a}_std": a_std,
                f"{label_b}_mean": b_mean,
                f"{label_b}_std": b_std,
                f"delta_{label_b}_minus_{label_a}": delta,
                f"pct_change_vs_{label_a}": pct_change_vs_a,
                "improvement_pct": improvement_pct,
                "better": better,
                "verdict": verdict,
            }
        )

    settings_rows = []
    for key in OBJECTIVE_KEYS:
        settings_rows.append(
            {
                "setting": key,
                label_a: objective_a.get(key, "<missing>"),
                label_b: objective_b.get(key, "<missing>"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    settings_df = pd.DataFrame(settings_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_safe_label(label_a)}_vs_{_safe_label(label_b)}"
    summary_csv = out_dir / f"{prefix}_results_summary.csv"
    summary_md = out_dir / f"{prefix}_results_summary.md"
    settings_csv = out_dir / f"{prefix}_objective_settings.csv"
    settings_md = out_dir / f"{prefix}_objective_settings.md"

    summary_df.to_csv(summary_csv, index=False)
    settings_df.to_csv(settings_csv, index=False)
    _write_markdown_table(summary_df, summary_md)
    _write_markdown_table(settings_df, settings_md)

    return {
        "summary_csv": summary_csv,
        "summary_md": summary_md,
        "settings_csv": settings_csv,
        "settings_md": settings_md,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved run results.csv summaries and objective settings.")
    parser.add_argument("--run-a", type=Path, required=True, help="Path to first run directory.")
    parser.add_argument("--run-b", type=Path, required=True, help="Path to second run directory.")
    parser.add_argument("--label-a", type=str, default="run_a", help="Label for first run.")
    parser.add_argument("--label-b", type=str, default="run_b", help="Label for second run.")
    parser.add_argument("--out-dir", type=Path, default=Path("model/saved/comparisons"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    outputs = compare_runs(args.run_a, args.run_b, args.label_a, args.label_b, args.out_dir)
    print("Wrote comparison files:")
    for path in outputs.values():
        print(path)
