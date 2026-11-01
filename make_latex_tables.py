"""Convert eval_boundary.py summary tables (record_*_full.txt) into a LaTeX table.

Why this exists: the record_*.txt files mix two row layouts (the baseline rows
are printed by parse_metrics, the DPRASR-Net row is pasted from DPRASRNet's own
report), so values are easy to mis-align when typed by hand. This script parses
them, prefers the authoritative "final test set results" block when the file
contains one, aligns every row to a single metric order, sorts the rows by Dice
and bolds the best value of each metric automatically.

Usage
-----
    python make_latex_tables.py                       # uses the default file names
    python make_latex_tables.py a.txt b.txt -o sota.tex
    python make_latex_tables.py --metrics Dice HD95 --layout matrix

Layouts
-------
    long    -- one row per (dataset, method); the datasets are stacked vertically.
    matrix  -- one row per method and one column group per dataset, so every method
               appears exactly once. With only two metrics per dataset this fits a
               single column; use it when the long table is too wide.

The dataset name is taken from the "dataset: XXX" header, falling back to the
file name. Missing values are printed as "-" and a warning is emitted whenever
the pasted row disagrees with the authoritative block, which is exactly the kind
of silent corruption this script is meant to catch.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# ---------------------------------------------------------------- constants
METRICS = ["Dice", "IoU", "ACC", "Precision", "Recall", "HD95", "ASSD", "NSD"]
LOWER_IS_BETTER = {"HD95", "ASSD"}
DECIMALS = {m: 2 for m in ("HD95", "ASSD")}

# order used by the summary table header printed by parse_metrics()
HEADER_ORDER = ["Dice", "ACC", "Recall", "IoU", "Precision", "HD95", "ASSD", "NSD"]
# order used when a row was pasted from DPRASRNet's own report (it contains "px")
PASTED_ORDER = ["Dice", "IoU", "ACC", "Precision", "Recall", "HD95", "ASSD", "NSD"]

DISPLAY = {
    "unet": "U-Net",
    "hiformer": "HiFormer",
    "missformer": "MISSFormer",
    "mkunet": "MK-UNet",
    "cmunet": "CMU-Net",
    "dbfnet": "DBF-Net",
    "dprasrnet": "DPRASR-Net (Ours)",
    "dprasr-net": "DPRASR-Net (Ours)",
}

DATASET_FROM_FILE = [
    ("bus-uclm", "BUS-UCLM"),
    ("bus_uclm", "BUS-UCLM"),
    ("busi", "BUSI"),
    ("udiat", "UDIAT"),
]


def canon_model(name: str) -> str:
    key = re.sub(r"[^a-z0-9-]", "", name.lower())
    return DISPLAY.get(key, name)


def dataset_name(path: str, text: str) -> str:
    m = re.search(r"dataset:\s*([A-Za-z0-9\-_]+)", text)
    if m:
        tag = m.group(1).lower()
        for needle, label in DATASET_FROM_FILE:
            if needle in tag:
                return label
        return m.group(1)
    base = os.path.basename(path).lower()
    for needle, label in DATASET_FROM_FILE:
        if needle in base:
            return label
    return os.path.basename(path)


def numbers(tokens: list[str]) -> list[float]:
    out = []
    for t in tokens:
        t = t.replace("px", "").replace("|", "").strip()
        if not t:
            continue
        try:
            out.append(float(t))
        except ValueError:
            continue
    return out


def parse_detail_block(text: str) -> dict[str, float] | None:
    """Parse DPRASRNet's own report, which is the authoritative source."""
    if "final test set results" not in text.lower():
        return None
    vals: dict[str, float] = {}
    patterns = {
        "Dice": r"Dice\s*:\s*([0-9.]+)",
        "IoU": r"IoU\s*:\s*([0-9.]+)",
        "ACC": r"ACC\s*:\s*([0-9.]+)",
        "Precision": r"Precision\s*:\s*([0-9.]+)",
        "Recall": r"Recall\s*:\s*([0-9.]+)",
        "HD95": r"HD95\s*:\s*([0-9.]+)",
        "ASSD": r"ASSD\s*:\s*([0-9.]+)",
        "NSD": r"NSD\s*:\s*([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            vals[key] = float(m.group(1))
    return vals or None


def parse_rows(text: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or set(s) <= {"=", "-"}:
            continue
        if s.startswith(("Model ", "Evaluation summary", "Generated:", "🏆", "🔹", "⚡", "⏱️", "🚀")):
            continue
        tokens = s.split()
        if len(tokens) < 5:
            continue
        name = canon_model(tokens[0])
        if name not in DISPLAY.values():
            continue
        vals = numbers(tokens[1:])
        if len(vals) < 8:
            continue
        order = PASTED_ORDER if ("px" in s or len(vals) < 10) else HEADER_ORDER
        rows[name] = {
            m: v for m, v in zip(order[: len(vals)], vals)
        }
    return rows


def fmt(value: float, metric: str) -> str:
    return f"{value:.{DECIMALS.get(metric, 4)}f}"


def latex_table(
    records: dict[str, dict[str, dict[str, float]]],
    metrics: list[str],
    caption: str,
    label: str,
) -> str:
    col_spec = "l|" + "".join("c" * len(metrics))
    head = " & ".join(
        m + (r"$\downarrow$" if m in LOWER_IS_BETTER else r"$\uparrow$") for m in metrics
    )
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{ll|{'c' * len(metrics)}}}",
        r"\hline",
        r"Dataset & Method & " + head + r" \\",
        r"\hline",
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ]

    body: list[str] = []
    refs: dict[str, dict[str, float]] = {}
    ours_name = "DPRASR-Net (Ours)"
    for ds, models in records.items():
        # baselines sorted by Dice (descending), our model kept as the last row
        ranked = sorted(
            (kv for kv in models.items() if kv[0] != ours_name),
            key=lambda kv: -kv[1].get("Dice", -1),
        )
        if ours_name in models:
            ranked.append((ours_name, models[ours_name]))
        n = len(ranked)
        refs[ds] = {}
        for m in metrics:
            present = [r[m] for r in models.values() if m in r]
            if present:
                refs[ds][m] = min(present) if m in LOWER_IS_BETTER else max(present)
        for i, (name, row) in enumerate(ranked):
            cells = []
            for m in metrics:
                if m not in row:
                    cells.append("-")
                    continue
                txt = fmt(row[m], m)
                best = refs[ds].get(m)
                if best is not None and abs(row[m] - best) < 5e-5:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
            first = rf"\multirow{{{n}}}{{*}}{{{ds}}}" if i == 0 else ""
            body.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
        body.append(r"\hline")

    # the last "\hline" duplicates the one that closes the header
    if body and body[-1] == r"\hline":
        body.pop()

    # splices the body between the two "\hline" markers of the skeleton
    out = lines[:]
    insert_at = 10  # index of the closing "\hline" of the header
    assert out[insert_at] == r"\hline", out[insert_at]
    out[insert_at:insert_at] = body
    return "\n".join(out)


def latex_table_matrix(records, metrics, caption, label, single_column=True) -> str:
    """One row per method, one column group per dataset (each method appears once)."""
    ours_name = "DPRASR-Net (Ours)"
    datasets = list(records.keys())

    names: list[str] = []
    for models in records.values():
        for n in models:
            if n not in names:
                names.append(n)

    def mean_dice(n: str) -> float:
        vals = [m[n]["Dice"] for m in records.values() if n in m and "Dice" in m[n]]
        return sum(vals) / len(vals) if vals else -1.0

    ranked = sorted((n for n in names if n != ours_name), key=mean_dice)
    ranked.reverse()
    if ours_name in names:
        ranked.append(ours_name)

    # best value of every metric inside each dataset
    refs = {}
    for ds, models in records.items():
        refs[ds] = {}
        for m in metrics:
            present = [r[m] for r in models.values() if m in r]
            if present:
                refs[ds][m] = min(present) if m in LOWER_IS_BETTER else max(present)

    head = " & ".join(
        m + (r"$\downarrow$" if m in LOWER_IS_BETTER else r"$\uparrow$") for m in metrics
    )
    metric_row = " & ".join([head] * len(datasets))
    group_heads = " & ".join(
        rf"\multicolumn{{{len(metrics)}}}{{c}}{{{ds}}}" for ds in datasets
    )
    rule = "".join(r"|" + "c" * len(metrics) for _ in datasets)

    env = "table" if single_column else "table*"
    width = r"\columnwidth" if single_column else r"\textwidth"

    lines = [
        rf"\begin{{{env}}}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{l{rule}}}",
        r"\hline",
        rf"& {group_heads} \\",
        rf"Method & {metric_row} \\",
        r"\hline",
    ]

    body: list[str] = []
    for name in ranked:
        cells = []
        for ds in datasets:
            row = records[ds].get(name, {})
            for m in metrics:
                if m not in row:
                    cells.append("-")
                    continue
                txt = fmt(row[m], m)
                best = refs[ds].get(m)
                if best is not None and abs(row[m] - best) < 5e-5:
                    txt = r"\textbf{" + txt + "}"
                cells.append(txt)
        body.append(f"{name} & " + " & ".join(cells) + r" \\")

    out = lines + body + [r"\hline", r"\end{tabular}%", r"}", rf"\end{{{env}}}"]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="record_*_full.txt files")
    ap.add_argument("-o", "--out", default="", help="write the LaTeX table here")
    ap.add_argument("--metrics", nargs="+", default=METRICS,
                    help="subset of: " + " ".join(METRICS))
    ap.add_argument("--layout", choices=["long", "matrix"], default="long",
                    help="long: one row per (dataset, method); "
                         "matrix: one row per method, one column group per dataset")
    ap.add_argument("--wide", action="store_true",
                    help="matrix layout only: span both columns (table*)")
    ap.add_argument("--caption", default="Quantitative comparison with state-of-the-art "
                                       "breast ultrasound segmentation methods on three "
                                       "public datasets (best in \\textbf{bold}).")
    ap.add_argument("--label", default="tab:sota")
    args = ap.parse_args(argv)

    files = args.files or ["record_busi_full.txt", "record_udiat_full.txt", "record_BUS-UCLM_full.txt"]
    known = {m.lower(): m for m in METRICS}
    metrics, unknown = [], []
    for m in args.metrics:
        canonical = known.get(m.lower())
        if canonical is None:
            unknown.append(m)
        elif canonical not in metrics:
            metrics.append(canonical)
    if unknown:
        ap.error(f"unknown metric(s): {unknown}; choose from {METRICS}")

    records: dict[str, dict[str, dict[str, float]]] = {}
    for path in files:
        if not os.path.exists(path):
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        ds = dataset_name(path, text)
        rows = parse_rows(text)
        detail = parse_detail_block(text)
        ours = "DPRASR-Net (Ours)"
        if detail:
            pasted = rows.get(ours)
            if pasted:
                diff = {k: (pasted.get(k), v) for k, v in detail.items()
                        if k in pasted and abs(pasted[k] - v) > 5e-4}
                if diff:
                    print(f"[fix] {ds}: pasted DPRASR-Net row disagrees with the "
                          f"final-report block, using the block:", file=sys.stderr)
                    for k, (a, b) in diff.items():
                        print(f"       {k}: row={a} block={b}", file=sys.stderr)
            rows[ours] = detail
        if not rows:
            print(f"[skip] {path}: no parsable rows", file=sys.stderr)
            continue
        records[ds] = rows

    if not records:
        print("nothing to do", file=sys.stderr)
        return 1

    # keep the dataset order BUSI, UDIAT, BUS-UCLM as in the paper
    order = {"BUSI": 0, "UDIAT": 1, "BUS-UCLM": 2}
    records = dict(sorted(records.items(), key=lambda kv: order.get(kv[0], 99)))

    if args.layout == "matrix":
        table = latex_table_matrix(records, metrics, args.caption, args.label,
                                   single_column=not args.wide)
    else:
        table = latex_table(records, metrics, args.caption, args.label)
    print(table)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(table + "\n")
        print(f"\nsaved -> {args.out}", file=sys.stderr)

    print("\n--- best per metric ---", file=sys.stderr)
    for ds, models in records.items():
        for m in metrics:
            present = [(n, r[m]) for n, r in models.items() if m in r]
            if not present:
                continue
            best = (min(present, key=lambda kv: kv[1]) if m in LOWER_IS_BETTER
                    else max(present, key=lambda kv: kv[1]))
            print(f"{ds:<9} {m:<9} {best[1]:.4f}  {best[0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
