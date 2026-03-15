import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def save_confusion_heatmap(confusion, output_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    positive_confusion = np.ma.masked_less_equal(confusion, 0)
    if positive_confusion.count() > 0:
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad(color="black")
        image = ax.imshow(
            positive_confusion,
            cmap=cmap,
            aspect="auto",
            norm=LogNorm(vmin=1e-3, vmax=float(positive_confusion.max())),
        )
        colorbar_label = "Mean count per fold (log scale)"
    else:
        image = ax.imshow(confusion, cmap="magma", aspect="auto")
        colorbar_label = "Mean count per fold"

    num_classes = confusion.shape[0]
    tick_positions = np.arange(num_classes)
    tick_labels = ["M"] + [str(idx) for idx in range(num_classes - 1)]

    ax.set_xlabel("Predicted fret class")
    ax.set_ylabel("True fret class")
    ax.set_title(title)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_epoch_labels(csv_dir):
    labels = set()
    for csv_path in csv_dir.glob("val_fret_confusion_epoch_*.csv"):
        try:
            labels.add(int(csv_path.stem.split("_")[-1]))
        except ValueError:
            continue
    return labels


def load_fold_confusions(run_dir, epoch):
    fold_dirs = sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No numeric fold directories found in {run_dir}.")

    matrices = []
    fold_names = []
    for fold_dir in fold_dirs:
        csv_path = fold_dir / "confusion" / "csv" / f"val_fret_confusion_epoch_{epoch:02d}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing confusion CSV for fold {fold_dir.name}: {csv_path}")
        frame = pd.read_csv(csv_path, index_col=0)
        matrices.append(frame)
        fold_names.append(fold_dir.name)
    return fold_names, matrices


def resolve_epoch(run_dir, requested_epoch):
    fold_dirs = sorted(
        [path for path in run_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    available_per_fold = []
    for fold_dir in fold_dirs:
        csv_dir = fold_dir / "confusion" / "csv"
        if not csv_dir.exists():
            raise FileNotFoundError(f"Missing confusion csv directory: {csv_dir}")
        labels = parse_epoch_labels(csv_dir)
        if not labels:
            raise FileNotFoundError(f"No confusion CSVs found in {csv_dir}")
        available_per_fold.append(labels)

    common_epochs = set.intersection(*available_per_fold)
    if not common_epochs:
        raise FileNotFoundError(f"No common confusion epoch found across folds in {run_dir}")

    if requested_epoch is not None:
        if requested_epoch not in common_epochs:
            raise ValueError(
                f"Requested epoch {requested_epoch} is not available across all folds. "
                f"Common epochs: {sorted(common_epochs)}"
            )
        return requested_epoch

    return max(common_epochs)


def main():
    parser = argparse.ArgumentParser(description="Average validation confusion matrices across folds.")
    parser.add_argument("--run-dir", required=True, help="Saved run directory containing fold confusion CSVs.")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number to average. Defaults to the latest epoch common to all folds.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    epoch = resolve_epoch(run_dir, args.epoch)
    fold_names, matrices = load_fold_confusions(run_dir, epoch)

    stacked = np.stack([frame.to_numpy(dtype=np.float64) for frame in matrices], axis=0)
    mean_confusion = stacked.mean(axis=0)

    index = matrices[0].index
    columns = matrices[0].columns
    out_dir = run_dir / "confusion" / "averaged"
    csv_path = out_dir / f"fold_avg_val_fret_confusion_epoch_{epoch:02d}.csv"
    png_path = out_dir / f"fold_avg_val_fret_confusion_epoch_{epoch:02d}.png"

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mean_confusion, index=index, columns=columns).to_csv(csv_path, float_format="%.6f")
    save_confusion_heatmap(
        mean_confusion,
        png_path,
        f"Fold-averaged validation fret confusion, epoch {epoch}",
    )

    print(f"Averaged folds: {', '.join(fold_names)}")
    print(f"Epoch: {epoch}")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
