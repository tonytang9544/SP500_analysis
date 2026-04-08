#!/usr/bin/env python3
"""Build commandlines to run inference and confusion for CSVs in data/with_stats.

Usage:
    python modeling/analyses/analysis.py           # prints commands (dry-run)
    python modeling/analyses/analysis.py --run     # actually execute commands

The script:
- collects all CSVs in `data/with_stats`
- constructs commands to call `modeling/inference.py` saving predictions to `results/`
- constructs commands to call `modeling/analyses/confusion.py` saving PNGs to `results/preds/`
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def find_csvs(data_dir: Path):
    if not data_dir.exists():
        return []
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def build_commands(csv_paths, results_dir: Path, preds_dir: Path, python_exe: str, extra_infer_args=None):
    extra_infer_args = extra_infer_args or []
    infer_cmds = []
    conf_cmds = []

    for csv in csv_paths:
        out_pred = results_dir / csv.name

        infer = [python_exe, "modeling/inference.py", "--csv_path", str(csv), "--out_csv", str(out_pred)]
        infer += extra_infer_args

        confusion_out = preds_dir / f"{csv.stem}_confusion.png"
        conf = [python_exe, "modeling/analyses/confusion.py", str(out_pred), str(csv), "--out", str(confusion_out)]

        infer_cmds.append(infer)
        conf_cmds.append(conf)

    return infer_cmds, conf_cmds


def print_cmd(cmd):
    print(shlex.join(cmd))


def run_cmd(cmd):
    print(f"Running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Construct/run inference and confusion commands for CSVs")
    p.add_argument("--data_dir", type=Path, default=Path("data/with_stats"))
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--preds_dir", type=Path, default=Path("results/preds"))
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    p.add_argument("--run", action="store_true", help="Execute the commands instead of printing (dry-run)")
    p.add_argument("--infer-arg", action="append", default=[], help="Additional argument(s) to append to inference command (can be repeated)")
    args = p.parse_args(argv)

    csvs = find_csvs(args.data_dir)
    if not csvs:
        print(f"No CSVs found in {args.data_dir}")
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.preds_dir.mkdir(parents=True, exist_ok=True)

    infer_cmds, conf_cmds = build_commands(csvs, args.results_dir, args.preds_dir, args.python, extra_infer_args=args.infer_arg)

    # print all commands first
    print("# Inference commands:")
    for c in infer_cmds:
        print_cmd(c)

    print("\n# Confusion commands:")
    for c in conf_cmds:
        print_cmd(c)

    if args.run:
        failed = []
        for c in infer_cmds:
            try:
                run_cmd(c)
            except subprocess.CalledProcessError as e:
                print(f"Inference failed: {e}")
                failed.append((c, e))

        for c in conf_cmds:
            try:
                run_cmd(c)
            except subprocess.CalledProcessError as e:
                print(f"Confusion plotting failed: {e}")
                failed.append((c, e))

        if failed:
            print(f"Finished with {len(failed)} failures")
            sys.exit(1)


if __name__ == "__main__":
    main()

