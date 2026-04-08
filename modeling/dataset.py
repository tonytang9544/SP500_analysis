import glob
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SP500SequenceDataset(Dataset):
    """Create sliding windows of length `seq_len` from CSVs in `data/with_stats`.

    Each sample x is a flattened tensor of shape (seq_len * num_features,),
    and y is the numeric stats of the next row (Open_change..log_market_cap).
    Date is transformed into cyclical features (day-of-week and day-of-year).
    Missing numeric values are forward/back filled then replaced with column medians.
    The dataset computes mean/std on the training data and applies simple standardization.
    """

    def __init__(self, data_dir="data/with_stats", seq_len=20, split="all", train_frac=0.8, val_frac=0.1, test_frac=0.1, columns=None):
        """Initialize dataset.

        split: one of 'train', 'val', 'test', or 'all'. When not 'all', windows
        are assigned per-file by the target row index so the splits are
        sequential (no leakage): train = first 80% rows, val = next 10%,
        test = last 10% (defaults to 0.8/0.1/0.1).
        """
        self.seq_len = seq_len
        self.split = split
        if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
            raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac

        # deterministic file order
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        self.samples = []  # list of (x_array, y_array)

        numeric_cols = columns if columns is not None else [
            "Open_change",
            "High_change",
            "Low_change",
            "Close_change",
            "exchange_portion",
            # "volatility",
            # "log_market_cap",
        ]

        self.output_dim = len(numeric_cols)

        for f in self.files:
            df = pd.read_csv(f, parse_dates=["date"])
            if df.empty:
                continue
            df = df.sort_values("date").reset_index(drop=True)

            # ensure numeric columns
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                else:
                    df[c] = np.nan

            # fill numeric NaNs robustly
            df[numeric_cols] = df[numeric_cols].ffill().bfill()
            df[numeric_cols] = df[numeric_cols].fillna(0.0)

            arr = df[numeric_cols].values.astype(np.float32)

            n = len(df)
            if n <= seq_len:
                continue

            # split boundaries are defined on row indices (0-based)
            train_end = int(n * self.train_frac)
            val_end = int(n * (self.train_frac + self.val_frac))

            # build sliding windows: x = rows[i:i+seq_len], y = row[i+seq_len]
            for i in range(n - seq_len - 1):
                y_idx = i + seq_len + 1 # target is the row after the next row of the input window
                x = arr[i : i + seq_len]
                y = arr[y_idx][: self.output_dim]

                if self.split in (None, "all"):
                    self.samples.append((x, y))
                elif self.split == "train":
                    if y_idx < train_end:
                        self.samples.append((x, y))
                    else:
                        break
                elif self.split == "val":
                    if y_idx >= train_end: 
                        if y_idx < val_end:
                            self.samples.append((x, y))
                        else:
                            break
                elif self.split == "test":
                    if y_idx >= val_end:
                        self.samples.append((x, y))
                else:
                    raise ValueError(f"unknown split: {self.split}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={self.split} in {data_dir} — check CSV files and seq_len.")


        # # compute feature-wise mean/std over all timesteps in the dataset
        # all_x = np.concatenate([s[0].reshape(-1, s[0].shape[-1]) for s in self.samples], axis=0)
        # self.mean = all_x.mean(axis=0)
        # self.std = all_x.std(axis=0) + 1e-9

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        # print(f"Sample {idx}: x shape={x.shape}, y shape={y.shape}")
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

