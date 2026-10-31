# -*- coding: utf-8 -*-
"""eval_boundary.py - add HD95 / ASSD / NSD to already-trained baselines.

It does NOT train anything. For every baseline that already has a saved
``best_*.pth``, it re-runs only the test phase of the baseline script
(same seed, same split, same checkpoint) and recomputes the full metric set,
including the boundary-sensitive metrics the paper reports.

Output:
    record_<stage>_full.txt   (8-metric table, one per stage)
    logs/<stage>_<model>_eval.log

Usage:
    python eval_boundary.py all
    python eval_boundary.py busi
    python eval_boundary.py udiat --models unet DBFNet

Notes:
    * DPRASRNet is skipped: its main.py already prints HD95/ASSD/NSD.
    * Baselines without a checkpoint (e.g. a crashed run) are marked failed.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from run_all import MODELS, STAGES, fmt, parse_metrics

REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)

MARKER = "@@BOUNDARY@@"
SKIP = {"DPRASRNet"}  # prints all eight metrics itself, no need to re-run.


def _merge_boundary(metrics, output):
    m = re.search(MARKER + r"(\{.*?\})", output)
    if not m:
        return metrics
    try:
        import json

        b = json.loads(m.group(1))
    except Exception:
        return metrics
    for k in ("hd95", "assd", "nsd"):
        v = b.get(k)
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v:  # nan
            continue
        metrics[k] = f"{v:.4f}" if k == "nsd" else f"{v:.2f}"
    metrics["n_boundary"] = b.get("n")
    return metrics


def eval_one(model, script, stage_name, idx, total):
    model_dir = REPO / model
    target = model_dir / script
    log_path = LOG_DIR / f"{stage_name}_{model}_eval.log"

    print(f"\n{'=' * 70}")
    print(f"▶ [{stage_name}] [{idx}/{total}] {model} -> {script} (TEST ONLY)")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    if not target.exists():
        print(f"  ❌ script missing: {target}")
        return {}, "script missing"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    import subprocess

    lines = []
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "_eval_runner.py"), script],
        cwd=str(model_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    proc.wait()

    output = "".join(lines)
    log_path.write_text(output, encoding="utf-8")

    metrics = _merge_boundary(parse_metrics(output), output)
    if proc.returncode != 0 or not metrics.get("dice"):
        tail = "\n".join(output.splitlines()[-15:])
        print(f"\n  ❌ failed(rc={proc.returncode})\n--- output tail ---\n{tail}")
        return metrics, f"failed(rc={proc.returncode})"

    print(
        f"\n  ✅ success | Dice={fmt(metrics.get('dice'))} "
        f"HD95={fmt(metrics.get('hd95'))} ASSD={fmt(metrics.get('assd'))} "
        f"NSD={fmt(metrics.get('nsd'))}"
    )
    return metrics, "success"


def run_stage(stage_name, file_idx, only=None):
    total = len(MODELS)
    results = []
    for i, (model, busi_f, udiat_f, busuclm_f) in enumerate(MODELS, 1):
        if model in SKIP:
            print(f"\n  (skip {model}: already reports HD95/ASSD/NSD)")
            continue
        if only and model not in only:
            continue
        script = [busi_f, udiat_f, busuclm_f][file_idx]
        metrics, status = eval_one(model, script, stage_name, i, total)
        results.append((model, metrics, status))

    lines = ["=" * 130,
             f"Evaluation summary table (8 metrics) - dataset: {stage_name}",
             f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 130]
    header = (
        f"{'Model':<16}{'Dice':<10}{'ACC':<10}{'Recall':<10}"
        f"{'IoU':<10}{'Precision':<12}{'HD95':<10}{'ASSD':<10}{'NSD':<10}"
        f"{'Reasoning ms':<14}{'FPS':<10}{'state':<14}"
    )
    lines.append(header)
    lines.append("-" * 130)
    for model, m, status in results:
        lines.append(
            f"{model:<16}{fmt(m.get('dice')):<10}{fmt(m.get('acc')):<10}"
            f"{fmt(m.get('recall')):<10}{fmt(m.get('iou')):<10}"
            f"{fmt(m.get('precision')):<12}"
            f"{fmt(m.get('hd95')):<10}{fmt(m.get('assd')):<10}"
            f"{fmt(m.get('nsd')):<10}"
            f"{fmt(m.get('infer_ms')):<14}{fmt(m.get('fps')):<10}"
            f"{status:<14}"
        )
    lines.append("=" * 130 + "\n")

    out = REPO / f"record_{stage_name}_full.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    ok = sum(1 for *_, s in results if s == "success")
    print(f"\nStage [{stage_name}] done: {ok}/{len(results)} success -> {out}")


def main():
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    only = None
    if "--models" in sys.argv:
        only = set(sys.argv[sys.argv.index("--models") + 1:])

    stage_map = {s[0].lower(): s for s in STAGES}
    if arg == "all":
        stages = STAGES
    elif arg in stage_map:
        stages = [stage_map[arg]]
    else:
        print(f"Unknown stage: {arg} | choose from all / {' / '.join(s[0].lower() for s in STAGES)}")
        sys.exit(1)

    for stage_name, _record, file_idx in stages:
        run_stage(stage_name, file_idx, only)


if __name__ == "__main__":
    main()
