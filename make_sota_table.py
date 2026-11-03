#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_sota_table.py
============================================================
Merge run_all.py summary tables (one or more txt files, possibly with
repeated dataset blocks from re-runs) into ONE LaTeX comparison table
with the three datasets as column groups.

Input line format (as written by run_all.py):
    Model  Dice ACC Recall IoU Precision HD95 ASSD NSD  ms  FPS  state
    unet   0.5111  ...                                   success
    unet   - - - - - - - - - -                           failed(rc=1)

Blocks may be repeated; later occurrences override earlier ones
(latest run wins), and a block only updates the models it contains.

Usage (repo root):
    python make_sota_table.py summary.txt
    python make_sota_table.py summary1.txt summary2.txt -o sota_table.tex
    python make_sota_table.py summary.txt --ours ours.json
    python make_sota_table.py summary.txt --metrics Dice,IoU,HD95,ASSD,NSD
    python make_sota_table.py summary.txt --layout long   # one block per dataset
    python make_sota_table.py summary.txt --skip-missing  # hide models w/o any data

The DPRASR-Net (Ours) row is taken from --ours (a small json file):
    {"BUSI":     {"Dice": 0.8177, "IoU": 0.7353, "HD95": 20.75, "ASSD": 8.39, "NSD": 0.5405},
     "UDIAT":    {...}, "BUS-UCLM": {...}}
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------- constants

DATASETS = ["BUSI", "UDIAT", "BUS-UCLM"]

DS_ALIASES = {
    "busi": "BUSI", "busi.txt": "BUSI",
    "udiat": "UDIAT", "udiat.txt": "UDIAT",
    "uclm": "BUS-UCLM", "bus-uclm": "BUS-UCLM", "bus_uclm": "BUS-UCLM",
    "bus-uclm.txt": "BUS-UCLM",
}

# display order = order of the paper's SOTA table; Ours always last
MODEL_ORDER = ["HiFormer", "CMU-Net", "MISSFormer", "DBF-Net",
               "MK-UNet", "U-Net", "DPRASR-Net"]
MODEL_ALIASES = {
    "unet": "U-Net", "u-net": "U-Net",
    "hiformer": "HiFormer",
    "missformer": "MISSFormer",
    "mkunet": "MK-UNet", "mk-unet": "MK-UNet",
    "cmunet": "CMU-Net", "cmu-net": "CMU-Net",
    "dbfnet": "DBF-Net", "dbf-net": "DBF-Net",
    "dprasrnet": "DPRASR-Net", "dprasr-net": "DPRASR-Net", "dprasr": "DPRASR-Net",
}

# column order inside a run_all summary row
ROW_COLS = ["Dice", "ACC", "Recall", "IoU", "Precision",
            "HD95", "ASSD", "NSD", "ms", "FPS"]

LOWER_IS_BETTER = {"HD95", "ASSD", "ms"}
TWO_DECIMALS = {"HD95", "ASSD", "ms", "FPS"}

ARROW = {"Dice": r"Dice$\uparrow$", "IoU": r"IoU$\uparrow$",
         "ACC": r"ACC$\uparrow$", "Recall": r"Recall$\uparrow$",
         "Precision": r"Precision$\uparrow$", "NSD": r"NSD$\uparrow$",
         "HD95": r"HD95$\downarrow$", "ASSD": r"ASSD$\downarrow$",
         "ms": r"ms$\downarrow$", "FPS": r"FPS$\uparrow$"}


# ------------------------------------------------------------------ parsing

def norm_ds(token):
    return DS_ALIASES.get(token.strip().lower())


def norm_model(token):
    return MODEL_ALIASES.get(token.strip().lower())


def parse_summary(path, records, overrides):
    """Append (ds, model) -> values entries found in `path` into `records`."""
    n_new = 0
    cur_ds = None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "data set:" in low:                       # block header line
            cur_ds = norm_ds(line.split("data set:")[-1])
            continue
        tokens = line.split()
        if len(tokens) == 1:                         # orphan line like "uclm"
            ds = norm_ds(tokens[0])
            if ds:
                cur_ds = ds
            continue
        if tokens[0].lower().startswith("model"):    # table header
            continue
        if set(line) <= {"=", "-"}:                  # separator
            continue
        model = norm_model(tokens[0])
        if model is None or cur_ds is None:
            continue
        state = tokens[-1].lower()
        if state.startswith("failed"):
            continue                                 # no numbers on this row
        nums = tokens[1:-1] if state.startswith("success") else tokens[1:]
        if len(nums) != len(ROW_COLS):
            continue
        values = {}
        for name, tok in zip(ROW_COLS, nums):
            try:
                values[name] = float(tok)
            except ValueError:
                values[name] = None
        key = (cur_ds, model)
        if key in records:
            overrides.append((path, cur_ds, model))
        records[key] = values
        n_new += 1
    return n_new


def load_ours(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for ds, metrics in data.items():
        key = norm_ds(ds) or ds
        out[(key, "DPRASR-Net")] = {k: float(v) for k, v in metrics.items()}
    return out


# ------------------------------------------------------------------- output

def fmt(metric, value):
    if value is None:
        return "--"
    return f"{value:.2f}" if metric in TWO_DECIMALS else f"{value:.4f}"


def best_formatted(metric, cells):
    """Return the set of formatted strings that deserve bold."""
    vals = [c for c in cells if c != "--"]
    if not vals:
        return set()
    nums = [float(c) for c in vals]          # compare numerically, not as strings
    target = min(nums) if metric in LOWER_IS_BETTER else max(nums)
    return {c for c, n in zip(vals, nums) if n == target}


def render_matrix(records, metrics, wide, skip_missing, ours_missing):
    lines = []
    env = "table*" if wide else "table"
    box = r"\resizebox{\textwidth}{!}{%}" if wide else r"\resizebox{\columnwidth}{!}{%}"
    lines += [
        rf"\begin{{{env}}}[!t]",
        r"\centering",
        r"\caption{Quantitative comparison with state-of-the-art breast ultrasound "
        r"segmentation methods on three public datasets (best in \textbf{bold}).}",
        r"\label{tab:sota_merged}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        box,
    ]
    ncol = 1 + len(metrics) * len(DATASETS)
    lines.append(r"\begin{tabular}{" + "l" + "|" + "|".join("c" * len(metrics) for _ in DATASETS) + "}")
    lines.append(r"\hline")
    # header row 1: dataset group names
    head1 = [""] + [rf"\multicolumn{{{len(metrics)}}}{{c}}{{{ds}}}" for ds in DATASETS]
    lines.append(" & ".join(head1) + r" \\")
    # header row 2: metric names
    head2 = ["Method"] + [ARROW[m] for _ in DATASETS for m in metrics]
    lines.append(" & ".join(head2) + r" \\")
    lines.append(r"\hline")

    shown = [m for m in MODEL_ORDER
             if not skip_missing or any((ds, m) in records for ds in DATASETS)]
    # pre-render cells so bolding can be decided per (ds, metric) column
    cells = {}
    for model in shown:
        for ds in DATASETS:
            v = records.get((ds, model), {})
            cells[(model, ds)] = [fmt(m, v.get(m)) for m in metrics]

    for model in shown:
        row = [model + " (Ours)" if model == "DPRASR-Net" else model]
        for ds in DATASETS:
            for i, m in enumerate(metrics):
                c = cells[(model, ds)][i]
                col = [cells[(mo, ds)][i] for mo in shown]
                row.append(rf"\textbf{{{c}}}" if c in best_formatted(m, col) else c)
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\hline", r"\end{tabular}%", "}", rf"\end{{{env}}}"]
    return "\n".join(lines)


def render_long(records, metrics, wide, skip_missing, ours_missing):
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Quantitative comparison on three public datasets (best in \textbf{bold}).}",
        r"\label{tab:sota_merged}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}", r"\hline",
        " & " + " & ".join(ARROW[m] for m in metrics) + r" \\", r"\hline",
    ]
    shown = [m for m in MODEL_ORDER
             if not skip_missing or any((ds, m) in records for ds in DATASETS)]
    for ds in DATASETS:
        lines.append(rf"\multicolumn{{{2 + len(metrics)}}}{{c}}{{\textbf{{{ds}}}}} \\")
        cells = {}
        for model in shown:
            v = records.get((ds, model), {})
            cells[model] = [fmt(m, v.get(m)) for m in metrics]
        for model in shown:
            row = [model + " (Ours)" if model == "DPRASR-Net" else model, ""]
            for i, m in enumerate(metrics):
                c = cells[model][i]
                col = [cells[mo][i] for mo in shown]
                row.append(rf"\textbf{{{c}}}" if c in best_formatted(m, col) else c)
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}%", "}", r"\end{table}"]
    return "\n".join(lines)


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="run_all summary txt file(s)")
    ap.add_argument("-o", "--out", default=None, help="write tex here (default: stdout)")
    ap.add_argument("--ours", default="ours.json",
                    help="json with the DPRASR-Net row (default: ours.json if present)")
    ap.add_argument("--metrics", default="Dice,IoU,HD95,ASSD,NSD",
                    help="comma list from Dice,IoU,ACC,Recall,Precision,HD95,ASSD,NSD,ms,FPS")
    ap.add_argument("--layout", choices=["matrix", "long"], default="matrix")
    ap.add_argument("--no-wide", action="store_true",
                    help="matrix layout as single-column table instead of table*")
    ap.add_argument("--skip-missing", action="store_true",
                    help="omit models that have no successful run at all")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in ROW_COLS]
    if bad:
        sys.exit(f"unknown metric(s): {bad}\nvalid: {ROW_COLS}")

    records, overrides = {}, []
    for path in args.inputs:
        n = parse_summary(path, records, overrides)
        print(f"[parse] {path}: {n} successful rows")
    for path, ds, model in overrides:
        print(f"[merge] {ds}/{model}: newer run in {Path(path).name} overrides older value")

    ours_missing = False
    ours_path = Path(args.ours)
    if ours_path.exists():
        records.update(load_ours(ours_path))
        print(f"[ours ] DPRASR-Net row loaded from {ours_path}")
    else:
        ours_missing = True
        print(f"[ours ] {ours_path} not found -> DPRASR-Net row omitted")

    # report gaps so nothing silently turns into "--"
    print("[gaps ]")
    any_gap = False
    for model in MODEL_ORDER:
        for ds in DATASETS:
            if (ds, model) not in records:
                print(f"         {ds}/{model}: no successful run (--)")
                any_gap = True
    if not any_gap:
        print("         none - table is complete")

    tex = (render_matrix if args.layout == "matrix" else render_long)(
        records, metrics, not args.no_wide if args.layout == "matrix" else False,
        args.skip_missing, ours_missing)

    if args.out:
        Path(args.out).write_text(tex + "\n", encoding="utf-8")
        print(f"[write] {args.out}")
    else:
        print()
        print(tex)


if __name__ == "__main__":
    main()
