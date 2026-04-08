#!/usr/bin/env python3
"""
Reconstruct price columns from predicted percent-change outputs.

Usage:
    python modeling/back_to_price.py --pred_csv modeling/out.csv --input_csv data/with_market_cap/XXX.csv --out_csv modeling/pred_prices.csv

The script looks for `Open_change`, `High_change`, `Low_change`, `Close_change` in the
predictions CSV (these are percent changes computed with `pct_change()` in preprocessing).
For each prediction row it finds the corresponding row in the original input CSV by date
or `y_idx` and uses the previous row's prices to compute predicted prices:
    predicted_price = previous_price * (1 + predicted_pct_change)

The output CSV will contain predicted Open/High/Low/Close plus the actual prices when
available for comparison.
"""
import argparse
import sys
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CHANGE_COLS = ["Open_change", "High_change", "Low_change", "Close_change"]
PRICE_COLS = ["Open", "High", "Low", "Close"]


def find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() == "date":
            return c
    return None


def load_preds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # try parse date column if present
    date_col = find_date_column(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.rename(columns={date_col: "date"}, inplace=True)
    return df


def load_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = find_date_column(df)
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df.rename(columns={date_col: "date"}, inplace=True)
    return df


def reconstruct_prices(preds: pd.DataFrame, original: pd.DataFrame, change_cols=DEFAULT_CHANGE_COLS):
    out_rows = []

    has_date = "date" in preds.columns and preds["date"].notna().any()
    for _, prow in preds.iterrows():
        row = {}
        row["y_idx"] = prow.get("y_idx")
        row["date"] = prow.get("date") if has_date else None

        # find the target row in original data
        target_idx = None
        if has_date and pd.notna(prow.get("date")):
            matches = original.index[original["date"] == pd.to_datetime(prow.get("date"))].tolist()
            if matches:
                target_idx = matches[0]
        # fallback to y_idx if available
        if target_idx is None and pd.notna(prow.get("y_idx")):
            try:
                yi = int(prow.get("y_idx"))
                # if y_idx is 0-based row index used by preprocessing, use it
                if 0 <= yi < len(original):
                    target_idx = yi
            except Exception:
                target_idx = None

        if target_idx is None:
            # cannot align, write NaNs
            for pc in PRICE_COLS:
                row[f"pred_{pc}"] = None
                row[f"actual_{pc}"] = None
            out_rows.append(row)
            continue

        # need previous row to compute price = prev_price * (1 + pct_change)
        if target_idx == 0:
            prev_row = None
        else:
            prev_row = original.iloc[target_idx - 1]

        # actual target row if available
        actual_row = original.iloc[target_idx] if target_idx < len(original) else None

        for ch_col, pc in zip(change_cols, PRICE_COLS):
            pred_change = prow.get(ch_col)
            try:
                pred_change = float(pred_change)
            except Exception:
                pred_change = None

            if prev_row is None or pred_change is None or pd.isna(pred_change):
                row[f"pred_{pc}"] = None
            else:
                prev_price = prev_row.get(pc)
                try:
                    prev_price = float(prev_price)
                except Exception:
                    prev_price = None

                if prev_price is None or pd.isna(prev_price):
                    row[f"pred_{pc}"] = None
                else:
                    row[f"pred_{pc}"] = prev_price * (1.0 + pred_change)

            if actual_row is None:
                row[f"actual_{pc}"] = None
            else:
                row[f"actual_{pc}"] = actual_row.get(pc)

        out_rows.append(row)

    return pd.DataFrame(out_rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True, help="Predictions CSV (e.g. modeling/out.csv)")
    parser.add_argument("--input_csv", required=True, help="Original input CSV with prices (Date/Open/High/Low/Close)")
    parser.add_argument("--out_csv", default=None, help="Path to write reconstructed prices CSV")
    args = parser.parse_args(argv)

    preds = load_preds(args.pred_csv)
    original = load_input(args.input_csv)

    # ensure expected price columns exist
    missing = [c for c in PRICE_COLS if c not in original.columns]
    if missing:
        print(f"Input CSV is missing expected price columns: {missing}", file=sys.stderr)
        print("Available columns:", list(original.columns), file=sys.stderr)
        sys.exit(2)

    reconstructed = reconstruct_prices(preds, original)

    if args.out_csv:
        reconstructed.to_csv(args.out_csv, index=False)
        print(f"Saved reconstructed prices to {args.out_csv}")
        return

    # Plot reconstructed vs actual prices over time
    # decide x-axis: prefer 'date' if available otherwise use y_idx or integer index
    if "date" in reconstructed.columns and reconstructed["date"].notna().any():
        x = pd.to_datetime(reconstructed["date"], errors="coerce")
        # ensure datetimes are tz-naive (matplotlib has trouble with some tz-aware timestamps)
        try:
            tzinfo = x.dt.tz
        except Exception:
            tzinfo = None
        if tzinfo is not None:
            try:
                x = x.dt.tz_convert("UTC").dt.tz_localize(None)
            except Exception:
                # fallback: convert to python datetimes without tz
                x = x.dt.tz_convert("UTC").dt.tz_localize(None)
        x_label = "date"
    elif reconstructed["y_idx"].notna().any():
        x = reconstructed["y_idx"]
        x_label = "y_idx"
    else:
        x = reconstructed.index
        x_label = "index"

    fig, axes = plt.subplots(nrows=len(PRICE_COLS), ncols=1, sharex=True, figsize=(10, 3 * len(PRICE_COLS)))
    if len(PRICE_COLS) == 1:
        axes = [axes]

    for ax, pc in zip(axes, PRICE_COLS):
        pred_col = f"pred_{pc}"
        act_col = f"actual_{pc}"
        if pred_col in reconstructed.columns:
            ax.plot(x, reconstructed[pred_col], label=f"pred_{pc}", marker="o", linestyle="-")
        if act_col in reconstructed.columns:
            ax.plot(x, reconstructed[act_col], label=f"actual_{pc}", marker=".", linestyle="--")
        ax.set_ylabel(pc)
        ax.legend()

    axes[-1].set_xlabel(x_label)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
