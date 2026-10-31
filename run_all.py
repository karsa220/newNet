# -*- coding: utf-8 -*-
from seed.set_seed import set_seed

"""run_all.py - DPRASR full model serial training automatic scheduling script
============================================================
Function:
  1. Run all model training in series, in order: first all models BUS-UCLM -> then all models udiat -> finally all models busi
  2. Console: Output all training logs and progress in real time (keep the output as is)
  3. Log file: The complete training log of each model is saved to logs/{stage}_{model}.log
  4. Final record file (record_*.txt): only records the summary table of all final models, and does not write the details of the intermediate process.
  5. Combined file (record_ALL.txt): merges every finished stage table into a single file.

Usage:
    python run_all.py # Run all three stages (BUS-UCLM -> udiat -> busi)
    python run_all.py bus-uclm # Only run the BUS-UCLM stage
    python run_all.py udiat # Only run the udiat phase
    python run_all.py busi # Only run the busi phase"""
import os
import re
import sys
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# DPRASR root directory (the directory where this script is located)
DPRASR = Path(__file__).resolve().parent
LOG_DIR = DPRASR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Model order and file mapping: (model folder, busi file, udiat file, BUS_UCLM file)
MODELS = [
    ("unet", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("unetplusplus", "main.py", "UDIAT.py", "BUS_UCLM.py"),
    ("hiformer", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("missformer", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("MKUnet", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("CMUNet", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("DPRASRNet", "main.py", "udiat.py", "BUS_UCLM.py"),
    ("DBFNet", "main.py", "udiat.py", "BUS_UCLM.py"),
]

# Three stages: (stage name, record file name, file index 0=busi 1=udiat 2=BUS-UCLM)
STAGES = [
    ("BUS-UCLM", "record_BUS-UCLM.txt", 2),
    ("udiat", "record_udiat.txt", 1),
    ("busi", "record_busi.txt", 0),
]


# -------------------------------------------------------------------
# Indicator analysis
# -------------------------------------------------------------------
def parse_metrics(output):
    """Parse test metrics from training output text."""
    idx = max(
        output.rfind("🏆"), output.rfind("Test set results"), output.rfind("final test")
    )
    seg = output[idx:] if idx >= 0 else output

    def find(pat):
        m = re.search(pat, seg)
        return m.group(1) if m else None

    return {
        "dice": find(r"Dice(?:\s*\(F1\))?\s*:?\s*([\d.]+)"),
        "iou": find(r"IoU\s*:?\s*([\d.]+)"),
        "acc": find(r"ACC\s*:?\s*([\d.]+)"),
        "precision": find(r"Pre(?:cision)?\s*:?\s*([\d.]+)"),
        "recall": find(r"Rec(?:all)?\s*:?\s*([\d.]+)"),
        "hd95": find(r"HD95\s*:?\s*([\d.]+)"),
        "assd": find(r"ASSD\s*:?\s*([\d.]+)"),
        "nsd": find(r"NSD\s*:?\s*([\d.]+)"),
        "infer_ms": find(
            r"(?:Average time consumption|Average time)\s*:?\s*([\d.]+)\s*ms"
        ),
        "fps": find(r"(?:F P S|FPS)\s*:?\s*([\d.]+)"),
    }


def fmt(v):
    """Format indicator value, None -> '-'"""
    return v if v else "-"

    # -------------------------------------------------------------------
    # Single model training
    # -------------------------------------------------------------------


def run_one(model_name, script_name, stage_name, idx, total):
    """Run a single model training script.
    The console displays the output in real time and saves it completely to logs/*.log.
    """
    model_dir = DPRASR / model_name
    script = model_dir / script_name
    log_path = LOG_DIR / f"{stage_name }_{model_name }.log"

    print(f"\n{'='*70 }")
    print(f"▶ [{stage_name }] [{idx }/{total }] {model_name } -> {script_name }")
    print(f"Working directory:{model_dir }")
    print(f"Log file:{log_path .relative_to (DPRASR )}")
    print(f"{'='*70 }")
    sys.stdout.flush()

    if not script.exists():
        msg = f"Script does not exist:{script }"
        print(f"  ❌ {msg }")
        log_path.write_text(msg, encoding="utf-8")
        return {}, "Script does not exist"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(DPRASR) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    lines = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script)],
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
            # Console real-time printout
            sys.stdout.write(line)
            sys.stdout.flush()

        proc.wait()
        output = "".join(lines)
        log_path.write_text(output, encoding="utf-8")

        metrics = parse_metrics(output)
        if proc.returncode != 0:
            status = f"failed(rc={proc .returncode })"
            tail = "\n".join(output.splitlines()[-25:])
            print(f"\n  ❌ {status }")
            print(
                f"---Output at the end ---\n{tail }\n --- See log for details:{log_path .name } ---"
            )
        else:
            status = "success"
            print(
                f"\n  ✅ {status } | Dice={fmt (metrics .get ('dice'))} "
                f"ACC={fmt (metrics .get ('acc'))} Recall={fmt (metrics .get ('recall'))} "
                f"FPS={fmt (metrics .get ('fps'))}"
            )
        return metrics, status
    except Exception as e:
        traceback.print_exc()
        output = "".join(lines) + f"\n[run_all exception]{e }"
        log_path.write_text(output, encoding="utf-8")
        return {}, f"abnormal:{e }"

        # -------------------------------------------------------------------
        # single stage processing
        # -------------------------------------------------------------------


def run_stage(stage_name, record_name, file_idx):
    record_path = DPRASR / record_name
    total = len(MODELS)
    results = []

    print(f"\n{'#'*70 }")
    print(
        f"# Stage:{stage_name }| Total{total }models | summary table ->{record_name }"
    )
    print(f"{'#'*70 }")
    sys.stdout.flush()

    for i, (model, busi_f, udiat_f, busuclm_f) in enumerate(MODELS, 1):
        files = [busi_f, udiat_f, busuclm_f]
        script = files[file_idx]
        metrics, status = run_one(model, script, stage_name, i, total)
        results.append((model, metrics, status))

        # Generate final summary table content
    table_lines = []
    table_lines.append("=" * 130)
    table_lines.append(f"Training record summary table - data set:{stage_name }")
    table_lines.append(
        f"Generation time:{datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}"
    )
    table_lines.append("=" * 130)
    header = (
        f"{'Model':<16}{'Dice':<10}{'ACC':<10}{'Recall':<10}"
        f"{'IoU':<10}{'Precision':<12}{'HD95':<10}{'ASSD':<10}{'NSD':<10}"
        f"{'Reasoning ms':<14}{'FPS':<10}{'state':<14}"
    )
    table_lines.append(header)
    table_lines.append("-" * 130)

    for model, m, status in results:
        row = (
            f"{model :<16}{fmt (m .get ('dice')):<10}{fmt (m .get ('acc')):<10}"
            f"{fmt (m .get ('recall')):<10}{fmt (m .get ('iou')):<10}"
            f"{fmt (m .get ('precision')):<12}"
            f"{fmt (m .get ('hd95')):<10}{fmt (m .get ('assd')):<10}"
            f"{fmt (m .get ('nsd')):<10}"
            f"{fmt (m .get ('infer_ms')):<14}"
            f"{fmt (m .get ('fps')):<10}{status :<14}"
        )
        table_lines.append(row)
    table_lines.append("=" * 130 + "\n")

    summary_text = "\n".join(table_lines)

    # Only the final table is written to record_*.txt files
    record_path.write_text(summary_text, encoding="utf-8")

    ok = sum(1 for *_, s in results if s == "success")
    print(
        f"\nStage [{stage_name }] Finish:{ok }/{total }Success | Log file:{record_path }"
    )
    sys.stdout.flush()


def write_combined_record():
    """Merge all finished per-stage record tables into record_ALL.txt."""
    parts = []
    missing = []
    for stage_name, record_name, _ in STAGES:
        p = DPRASR / record_name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").rstrip())
        else:
            missing.append(record_name)
    if not parts:
        return
    out = DPRASR / "record_ALL.txt"
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"\nCombined summary -> {out}")
    if missing:
        print(f"  (stages not run yet: {', '.join(missing)})")
    sys.stdout.flush()

    # -------------------------------------------------------------------
    # main entrance
    # -------------------------------------------------------------------


def main():
    set_seed(42)
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    stage_map = {s[0].lower(): s for s in STAGES}

    if arg == "all":
        stages = STAGES
    elif arg in stage_map:
        stages = [stage_map[arg]]
    else:
        print(f"Unknown stage parameters:{arg }")
        print(f"Optional: all /{' / '.join (s [0 ].lower ()for s in STAGES )}")
        sys.exit(1)

    print(f"Python : {sys .executable }")
    print(f"DPRASR  : {DPRASR }")
    print(f"log   :{LOG_DIR }")
    print(f"Stage:{[s [0 ]for s in stages ]}")
    print(f"Number of models:{len (MODELS )} ({', '.join (m [0 ]for m in MODELS )})")
    print(f"Start time:{datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    for stage_name, record_name, file_idx in stages:
        run_stage(stage_name, record_name, file_idx)

    write_combined_record()

    print(f"\n{'='*70 }")
    print("All done. Log files:")
    for s in stages:
        print(f"  {DPRASR /s [1 ]}")
    print(f"End time:{datetime .now ().strftime ('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
