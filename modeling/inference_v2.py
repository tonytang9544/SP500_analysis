#!/usr/bin/env python3
"""
Inference v2: build windows consistent with `SP500SequenceDataset` (target = row after next)
and save CSV with dates aligned to the predicted target rows.

Usage:
    python modeling/inference_v2.py --model_path modeling/artifacts/transformer_model.pth --csv_path data/with_stats/AAPL.csv --out_csv results/AAPL_preds_v2.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch


def build_windows_dataset_style(df, numeric_cols, seq_len, mode="all"):
    # ensure numeric columns exist and are numeric
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    arr = df[numeric_cols].values.astype(np.float32)
    n = len(arr)

    # dataset logic requires at least n > seq_len + 1 to produce any windows
    if n <= seq_len + 1:
        raise RuntimeError(f"CSV has {n} rows but need > seq_len+1 ({seq_len + 1}) to build windows")

    has_dates = "date" in df.columns

    if mode == "last":
        # last valid window used by SP500SequenceDataset corresponds to i = n - seq_len - 2
        last_i = n - seq_len - 2
        x = arr[last_i : last_i + seq_len]
        y_idx = last_i + seq_len + 1
        meta = [{"y_idx": int(y_idx), "date": (pd.NaT if not has_dates else df["date"].iloc[int(y_idx)])}]
        return torch.from_numpy(x).unsqueeze(0), meta, list(df[numeric_cols].columns)
    else:
        windows = []
        meta = []
        # loop exactly as in SP500SequenceDataset: for i in range(n - seq_len - 1)
        for i in range(n - seq_len - 1):
            x = arr[i : i + seq_len]
            y_idx = i + seq_len + 1
            windows.append(x)
            meta.append({"y_idx": int(y_idx), "date": (pd.NaT if not has_dates else df["date"].iloc[int(y_idx)])})
        return torch.from_numpy(np.stack(windows, axis=0)), meta, list(df[numeric_cols].columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="modeling/artifacts/transformer_model.pth", help="Path to trained model .pth file")
    parser.add_argument("--csv_path", type=str, required=True, help="Single CSV in data/with_stats to run inference on")
    parser.add_argument("--seq_len", type=int, default=None, help="Override seq_len if model has no attribute")
    parser.add_argument("--mode", type=str, default="all", choices=["last", "all"], help="'last' uses final dataset-valid window; 'all' runs all dataset-valid sliding windows")
    parser.add_argument("--out_csv", type=str, default=None, help="Path to write predictions CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model from {args.model_path} onto device {device}")
    model = torch.load(args.model_path, weights_only=False)
    model.to(device)
    model.eval()

    # infer shape
    if hasattr(model, "seq_len") and hasattr(model, "num_features"):
        seq_len = model.seq_len
        num_features = model.num_features
    else:
        if args.seq_len is None:
            # try to deduce from linear layers if present
            try:
                in_features = model.linear.in_features
                out_features = model.output.out_features
                seq_len = in_features // out_features
                num_features = out_features
            except Exception:
                raise RuntimeError("Unable to infer model input shape; provide --seq_len or a model with attributes")
        else:
            seq_len = args.seq_len

    numeric_cols = [
        "Open_change",
        "High_change",
        "Low_change",
        "Close_change",
        "exchange_portion",
        # "volatility",
        # "log_market_cap",
    ]

    df = pd.read_csv(args.csv_path, parse_dates=["date"]) if args.csv_path is not None else pd.DataFrame()
    x, meta, feature_names = build_windows_dataset_style(df, numeric_cols, seq_len, mode=args.mode)

    x = x.to(device)
    with torch.no_grad():
        out = model(x)

    out_np = out.cpu().numpy()
    n_out = out_np.shape[0]
    out_flat = out_np.reshape(n_out, -1)

    if out_flat.shape[1] == len(feature_names):
        cols = feature_names
    else:
        cols = [f"out_{i}" for i in range(out_flat.shape[1])]

    out_df = pd.DataFrame(out_flat, columns=cols)
    out_df["y_idx"] = [m.get("y_idx") for m in meta]
    out_df["date"] = [m.get("date") for m in meta]

    print("Input windows:", x.shape)
    print("Output shape:", out.shape)

    if args.out_csv is not None:
        out_df.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to {args.out_csv}")
    else:
        print(out_df.head())


if __name__ == "__main__":
    main()
