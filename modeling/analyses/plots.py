#!/usr/bin/env python3
"""Plot confusion matrices comparing predicted vs actual price changes.

Usage:
    python modeling/analyses/confusion.py preds.csv truth.csv --out confusion.png

The script aligns rows by `y_idx` (if present in both files) or by `date` (if present),
falls back to index alignment otherwise. It discretizes each change into three classes
('down', 'flat', 'up') using a small tolerance and then plots a 2x2 grid of
confusion matrices for Open_change, High_change, Low_change, Close_change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_COLS = ["Open_change", "High_change", "Low_change", "Close_change"]


def to_label(val: float, tol: float) -> str:
    if pd.isna(val):
        return "nan"
    if val > tol:
        return "up"
    if val < -tol:
        return "down"
    return "flat"


def align_data(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    # prefer y_idx if present
    if "y_idx" in pred_df.columns and "y_idx" in truth_df.columns:
        merged = pd.merge(pred_df, truth_df, on="y_idx", suffixes=("_pred", "_truth"))
        merged.index = merged["y_idx"]
        return merged

    # else try date
    if "date" in pred_df.columns and "date" in truth_df.columns:
        a = pred_df.copy()
        b = truth_df.copy()
        a["date"] = pd.to_datetime(a["date"], errors="coerce")
        b["date"] = pd.to_datetime(b["date"], errors="coerce")
        merged = pd.merge(a, b, on="date", suffixes=("_pred", "_truth"))
        merged.index = merged["date"]
        return merged

    # fallback: align by row order (index)
    a = pred_df.reset_index(drop=True)
    b = truth_df.reset_index(drop=True)
    n = min(len(a), len(b))
    merged = pd.concat([a.iloc[:n].reset_index(drop=True), b.iloc[:n].reset_index(drop=True)], axis=1)
    return merged


def plot_confusion_matrices(merged: pd.DataFrame, cols=DEFAULT_COLS, tol=1e-3, out_path: Path | None = None):
    labels = ["down", "flat", "up"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, col in enumerate(cols):
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else col

        if pred_col not in merged.columns or truth_col not in merged.columns:
            axes[i].text(0.5, 0.5, f"{col} missing", ha="center", va="center")
            axes[i].set_title(col)
            axes[i].axis("off")
            continue

        pred_labels = merged[pred_col].apply(lambda v: to_label(v, tol))
        truth_labels = merged[truth_col].apply(lambda v: to_label(v, tol))

        # build confusion matrix via crosstab with fixed category order
        pred_cat = pd.Categorical(pred_labels, categories=labels, ordered=True)
        truth_cat = pd.Categorical(truth_labels, categories=labels, ordered=True)
        cm = pd.crosstab(truth_cat, pred_cat)

        # ensure all rows/cols present
        cm = cm.reindex(index=labels, columns=labels, fill_value=0)

        ax = axes[i]
        im = ax.imshow(cm.values, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(col)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        # annotate counts and percentages
        total = cm.values.sum()
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                count = int(cm.values[r, c])
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(c, r, f"{count}\n{pct:.1f}%", ha="center", va="center", color="black")

    fig.suptitle("Confusion matrices: predicted vs actual change direction")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved confusion matrices to {out_path}")


def simple_plot_over_history(merged: pd.DataFrame, cols=DEFAULT_COLS):
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3 * len(cols)))
    if len(cols) == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else col

        axes[i].plot(merged.index, merged[truth_col], label="Actual", color="blue")
        axes[i].plot(merged.index, merged[pred_col], label="Predicted", color="orange")
        axes[i].set_title(col)
        axes[i].legend()

    fig.suptitle("Predicted vs Actual Changes")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def scatter_pred_vs_actual(merged: pd.DataFrame, cols=DEFAULT_COLS):
    
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 3 * len(cols)))
    for i, col in enumerate(cols):
        pred_col = f"{col}_pred" if f"{col}_pred" in merged.columns else col
        truth_col = f"{col}_truth" if f"{col}_truth" in merged.columns else col

        if pred_col not in merged.columns or truth_col not in merged.columns:
            axes[i].text(0.5, 0.5, f"{col} missing", ha="center", va="center")
            axes[i].set_title(col)
            axes[i].axis("off")
            continue

        axes[i].scatter(merged[truth_col], merged[pred_col], alpha=0.5)
        axes[i].plot([merged[truth_col].min(), merged[truth_col].max()], [merged[truth_col].min(), merged[truth_col].max()], "r--")
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_title(f"Predicted vs Actual for {col}")
        axes[i].grid()

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual for {col}")
    plt.grid()
    plt.show()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot confusion matrices for predicted vs truth CSVs")
    parser.add_argument("pred_csv", help="Predictions CSV (from modeling/inference.py --out_csv)")
    parser.add_argument("truth_csv", help="Truth CSV with actual values")
    parser.add_argument("--cols", nargs="*", default=DEFAULT_COLS, help="Columns to compare")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerance for considering change as flat")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). If omitted, shows interactively")
    args = parser.parse_args(argv)

    pred_df = pd.read_csv(args.pred_csv)
    truth_df = pd.read_csv(args.truth_csv)

    merged = align_data(pred_df, truth_df)

    plot_confusion_matrices(merged, cols=args.cols, tol=args.tol, out_path=Path(args.out) if args.out else None)
    simple_plot_over_history(merged, cols=args.cols)
    scatter_pred_vs_actual(merged, cols=args.cols)

if __name__ == "__main__":
    main()
