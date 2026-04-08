#!/usr/bin/env python3
"""
Simple inference script: loads a saved PyTorch model and runs random inputs.

Usage example:
    python modeling/implement.py --model_path modeling/artifacts/transformer_model.pth --batch 4
"""
import argparse
import numpy as np
import pandas as pd
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="modeling/artifacts/transformer_model.pth")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for random inputs (ignored if --csv_path is provided)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default=None, help="Path to a single CSV file to run inference on")
    parser.add_argument("--mode", type=str, default="last", choices=["last", "all"], help="'last' uses the final window, 'all' runs sliding windows")
    parser.add_argument("--seq_len", type=int, default=None, help="Optional sequence length override for models without attributes")
    parser.add_argument("--out_csv", type=str, default=None, help="Path to write predictions CSV (optional)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading model from {args.model_path} onto device {device}")
    model = torch.load(args.model_path, weights_only=False)
    model.to(device)
    model.eval()

    # infer input shape: prefer attributes on Transformer model, fall back to Linear inspection
    if hasattr(model, "seq_len") and hasattr(model, "num_features"):
        seq_len = model.seq_len
        num_features = model.num_features
    else:
        # try to deduce from linear layer shapes for the simple linear model
        try:
            in_features = model.linear.in_features
            out_features = model.output.out_features
            seq_len = in_features // out_features
            num_features = out_features
        except Exception:
            raise RuntimeError(
                "Unable to infer model input shape. If your model is custom, provide a compatible model or update this script."
            )

    # If a CSV path is provided, load it and construct input windows
    numeric_cols = [
        "Open_change",
        "High_change",
        "Low_change",
        "Close_change",
        "exchange_portion",
        # "volatility",
        # "log_market_cap",
    ]

    def build_windows_from_csv(csv_path, seq_len, mode="last"):
        df = pd.read_csv(csv_path, parse_dates=["date"]) if csv_path is not None else pd.DataFrame()
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
        if n < seq_len:
            raise RuntimeError(f"CSV has only {n} rows, but seq_len={seq_len} is required")
        # prepare metadata for outputs: target row indices and dates (if present)
        has_dates = "date" in df.columns
        feature_names = df[numeric_cols].columns.tolist()
        if mode == "last":
            x = arr[-seq_len:]
            meta = [{"y_idx": n, "date": (pd.NaT if not has_dates else df["date"].iloc[-1])}]
            return torch.from_numpy(x).unsqueeze(0), meta, feature_names  # shape (1, seq_len, num_features)
        else:
            windows = [arr[i : i + seq_len] for i in range(n - seq_len)]
            meta = [
                {"y_idx": i + seq_len, "date": (pd.NaT if not has_dates else df["date"].iloc[i + seq_len])}
                for i in range(n - seq_len)
            ]
            return torch.from_numpy(np.stack(windows, axis=0)), meta, feature_names

    if args.csv_path is not None:
        if args.seq_len is not None:
            seq_len = args.seq_len
        # build input tensor(s) and metadata from CSV
        x, meta, feature_names = build_windows_from_csv(args.csv_path, seq_len, mode=args.mode)
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
        # attach metadata if present
        if meta is not None:
            out_df["y_idx"] = [m.get("y_idx") for m in meta]
            out_df["date"] = [m.get("date") for m in meta]

        print("CSV input shape:", x.shape)
        print("Output shape:", out.shape)
        print("Outputs (first rows):\n", out_np[: min(5, out_np.shape[0])])

        if args.out_csv is not None:
            out_df.to_csv(args.out_csv, index=False)
            print(f"Saved predictions to {args.out_csv}")
    else:
        x = torch.randn(args.batch, seq_len, num_features, device=device)
        with torch.no_grad():
            out = model(x)
        print("Input shape:", x.shape)
        print("Output shape:", out.shape)
        # print a small portion of outputs
        print("Outputs (first rows):\n", out.cpu().numpy()[: min(5, out.shape[0])])

        if args.out_csv is not None:
            out_np = out.cpu().numpy()
            n_out = out_np.shape[0]
            out_flat = out_np.reshape(n_out, -1)
            # no original feature names available for random inputs
            cols = [f"out_{i}" for i in range(out_flat.shape[1])]
            out_df = pd.DataFrame(out_flat, columns=cols)
            out_df["y_idx"] = list(range(n_out))
            out_df.to_csv(args.out_csv, index=False)
            print(f"Saved predictions to {args.out_csv}")


if __name__ == "__main__":
    main()
