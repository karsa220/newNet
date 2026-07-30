# -*- coding: utf-8 -*-
from seed.set_seed import set_seed
"""
run_all.py - DPRASR 全模型串行训练自动调度脚本
============================================================
功能：
  1. 串行运行所有模型训练，顺序：先所有模型 BUS-UCLM -> 再所有模型 udiat -> 最后所有模型 busi
  2. 控制台：实时输出所有训练日志和进度（保持原样输出）
  3. 日志文件：每个模型的完整训练日志保存到 logs/{阶段}_{模型}.log
  4. 最终记录文件（record_*.txt）：只记录最终所有模型的汇总表格，不写中间过程明细

用法：
    python run_all.py            # 跑全部三个阶段（BUS-UCLM -> udiat -> busi）
    python run_all.py bus-uclm   # 只跑 BUS-UCLM 阶段
    python run_all.py udiat      # 只跑 udiat 阶段
    python run_all.py busi       # 只跑 busi 阶段
"""
import os
import re
import sys
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

# DPRASR 根目录（本脚本所在目录）
DPRASR = Path(__file__).resolve().parent
LOG_DIR = DPRASR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 模型顺序与文件映射: (模型文件夹, busi文件, udiat文件, BUS_UCLM文件)
MODELS = [
  ("unet",          "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("unetplusplus",  "main.py", "UDIAT.py",          "BUS_UCLM.py"),
  ("swinunet",      "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("segformer",     "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("hiformer",      "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("missformer",    "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("LVUnet",        "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("MKUnet",        "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("CMUNet",        "main.py", "udiat.py",          "BUS_UCLM.py"),
  ("DPRASRNet",      "main.py", "udiat.py",          "BUS_UCLM.py"),
("TransUNet",      "main.py", "udiat.py",          "BUS_UCLM.py"),

]

# 三个阶段: (阶段名, record文件名, 文件索引 0=busi 1=udiat 2=BUS-UCLM)
STAGES = [
    ("BUS-UCLM", "record_BUS-UCLM.txt", 2),
    ("udiat",    "record_udiat.txt",    1),
    ("busi",     "record_busi.txt",     0),
]


# -------------------------------------------------------------------
# 指标解析
# -------------------------------------------------------------------
def parse_metrics(output):
    """从训练输出文本解析测试指标。"""
    idx = max(output.rfind("🏆"), output.rfind("测试集成绩"), output.rfind("最终测试"))
    seg = output[idx:] if idx >= 0 else output

    def find(pat):
        m = re.search(pat, seg)
        return m.group(1) if m else None

    return {
        "dice":      find(r"Dice(?:\s*\(F1\))?\s*:?\s*([\d.]+)"),
        "iou":       find(r"IoU\s*:?\s*([\d.]+)"),
        "acc":       find(r"ACC\s*:?\s*([\d.]+)"),
        "precision": find(r"Pre(?:cision)?\s*:?\s*([\d.]+)"),
        "recall":    find(r"Rec(?:all)?\s*:?\s*([\d.]+)"),
        "infer_ms":  find(r"平均耗时\s*:?\s*([\d.]+)\s*ms"),
        "fps":       find(r"(?:F P S|FPS)\s*:?\s*([\d.]+)"),
    }


def fmt(v):
    """格式化指标值，None -> '-'"""
    return v if v else "-"


# -------------------------------------------------------------------
# 单模型训练
# -------------------------------------------------------------------
def run_one(model_name, script_name, stage_name, idx, total):
    """运行单个模型训练脚本。
    控制台实时显示输出，同时完整保存到 logs/*.log。
    """
    model_dir = DPRASR / model_name
    script = model_dir / script_name
    log_path = LOG_DIR / f"{stage_name}_{model_name}.log"

    print(f"\n{'=' * 70}")
    print(f"▶ [{stage_name}] [{idx}/{total}] {model_name} -> {script_name}")
    print(f"  工作目录: {model_dir}")
    print(f"  日志文件: {log_path.relative_to(DPRASR)}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    if not script.exists():
        msg = f"脚本不存在: {script}"
        print(f"  ❌ {msg}")
        log_path.write_text(msg, encoding="utf-8")
        return {}, "脚本不存在"

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
            # 控制台实时打印输出
            sys.stdout.write(line)
            sys.stdout.flush()

        proc.wait()
        output = "".join(lines)
        log_path.write_text(output, encoding="utf-8")

        metrics = parse_metrics(output)
        if proc.returncode != 0:
            status = f"失败(rc={proc.returncode})"
            tail = "\n".join(output.splitlines()[-25:])
            print(f"\n  ❌ {status}")
            print(f"  --- 末尾输出 ---\n{tail}\n  --- 详见日志: {log_path.name} ---")
        else:
            status = "成功"
            print(f"\n  ✅ {status} | Dice={fmt(metrics.get('dice'))} "
                  f"ACC={fmt(metrics.get('acc'))} Recall={fmt(metrics.get('recall'))} "
                  f"FPS={fmt(metrics.get('fps'))}")
        return metrics, status
    except Exception as e:
        traceback.print_exc()
        output = "".join(lines) + f"\n[run_all 异常] {e}"
        log_path.write_text(output, encoding="utf-8")
        return {}, f"异常:{e}"


# -------------------------------------------------------------------
# 单阶段处理
# -------------------------------------------------------------------
def run_stage(stage_name, record_name, file_idx):
    record_path = DPRASR / record_name
    total = len(MODELS)
    results = []

    print(f"\n{'#' * 70}")
    print(f"# 阶段: {stage_name} | 共 {total} 个模型 | 汇总表格 -> {record_name}")
    print(f"{'#' * 70}")
    sys.stdout.flush()

    for i, (model, busi_f, udiat_f, busuclm_f) in enumerate(MODELS, 1):
        files = [busi_f, udiat_f, busuclm_f]
        script = files[file_idx]
        metrics, status = run_one(model, script, stage_name, i, total)
        results.append((model, metrics, status))

    # 生成最终汇总表格内容
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append(f"训练记录汇总表 - 数据集: {stage_name}")
    table_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    table_lines.append("=" * 100)
    header = (f"{'模型':<16}{'Dice':<10}{'ACC':<10}{'Recall':<10}"
              f"{'IoU':<10}{'Precision':<12}{'推理ms':<10}{'FPS':<10}{'状态':<14}")
    table_lines.append(header)
    table_lines.append("-" * 100)

    for model, m, status in results:
        row = (f"{model:<16}{fmt(m.get('dice')):<10}{fmt(m.get('acc')):<10}"
               f"{fmt(m.get('recall')):<10}{fmt(m.get('iou')):<10}"
               f"{fmt(m.get('precision')):<12}{fmt(m.get('infer_ms')):<10}"
               f"{fmt(m.get('fps')):<10}{status:<14}")
        table_lines.append(row)
    table_lines.append("=" * 100 + "\n")

    summary_text = "\n".join(table_lines)

    # 仅将最终表格写入 record_*.txt 文件
    record_path.write_text(summary_text, encoding="utf-8")

    ok = sum(1 for *_, s in results if s == "成功")
    print(f"\n阶段 [{stage_name}] 完成: {ok}/{total} 成功 | 记录文件: {record_path}")
    sys.stdout.flush()


# -------------------------------------------------------------------
# 主入口
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
        print(f"未知阶段参数: {arg}")
        print(f"可选: all / {' / '.join(s[0].lower() for s in STAGES)}")
        sys.exit(1)

    print(f"Python : {sys.executable}")
    print(f"DPRASR  : {DPRASR}")
    print(f"日志   : {LOG_DIR}")
    print(f"阶段   : {[s[0] for s in stages]}")
    print(f"模型数 : {len(MODELS)} ({', '.join(m[0] for m in MODELS)})")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout.flush()

    for stage_name, record_name, file_idx in stages:
        run_stage(stage_name, record_name, file_idx)

    print(f"\n{'=' * 70}")
    print("全部完成。记录文件：")
    for s in stages:
        print(f"  {DPRASR / s[1]}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()