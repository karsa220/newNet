import re


def clean_model_name(name):
    """清洗模型名称，去除 LaTeX 引用、加粗标签以及括号备注"""
    name = re.sub(r'\\cite\{.*?\}', '', name)
    name = re.sub(r'\\textbf\{(.*?)\}', r'\1', name)
    name = re.sub(r'\\underline\{(.*?)\}', r'\1', name)
    name = name.replace('(Ours)', '').replace('最终测试集成绩', '').replace('测试集最终成绩', '').strip()

    # 归一化命名
    n = name.lower()
    if 'cmu-net' in n or 'cmunet' in n: return "CMU-Net"
    if 'missformer' in n: return "MISSFormer"
    if 'segformer' in n: return "SegFormer"
    if 'mk-unet' in n or 'mkunet' in n: return "MK-UNet"
    if 'hiformer' in n: return "HiFormer"
    if 'lv-unet' in n or 'lvunet' in n: return "LV-UNet"
    if 'swin' in n: return "SwinUNet"
    if 'unet++' in n or 'unetplusplus' in n: return "UNet++"
    if 'unet' in n and 'mk' not in n and 'lv' not in n and 'swin' not in n: return "UNet"
    if 'dprasr' in n: return "DPRASR-Net"

    # 清理残余的括号和特殊字符
    name = re.sub(r'\(.*?\)', '', name).strip()
    return name


def clean_value(val_str):
    """从可能包含 LaTeX 语法的字符串中提取浮点数"""
    match = re.search(r'(\d+\.\d+|\d+)', str(val_str))
    return float(match.group(1)) if match else 0.0


def main():
    # 引用格式映射表
    cites = {
        "UNet": "\\cite{unet}", "MISSFormer": "\\cite{missformer}", "HiFormer": "\\cite{hiformer}",
        "SwinUNet": "\\cite{swinunet}", "LV-UNet": "\\cite{lvunet}", "UNet++": "\\cite{unetplusplus}",
        "MK-UNet": "\\cite{mkunet}", "SegFormer": "\\cite{segformer}", "CMU-Net": "\\cite{cmunet}"
    }

    datasets = {"BUSI": {}, "UCLM": {}, "UDIAT": {}}
    current_ds = "BUSI"  # 默认数据集上下文

    try:
        with open('record.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ 未找到 record.txt，请在同目录下创建并粘贴数据。")
        return

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 1. 动态推断当前属于哪个数据集 (每一行都实时检测)
        ds_match = re.search(r'(BUSI|UCLM|UDIAT)', line, re.IGNORECASE)
        if ds_match:
            current_ds = ds_match.group(1).upper()

        # 2. 识别并解析 LaTeX 表格格式
        if '&' in line and ('\\\\' in line or line.count('&') >= 5):
            parts = line.split('&')
            if len(parts) >= 6:
                model = clean_model_name(parts[0])
                metrics = [clean_value(p) for p in parts[1:6]]

                if sum(metrics) > 0 and model:
                    # 靠后出现的最新数据直接覆盖旧数据
                    datasets[current_ds][model] = metrics

        # 3. 识别并解析终端日志格式 (🏆 开头)
        elif line.startswith('🏆'):
            header = line
            body = ""
            j = i + 1

            # 仅提取包含指标的行
            while j < len(lines):
                next_line = lines[j].strip()
                if (next_line.startswith('🏆') or next_line.startswith('=')
                        or next_line.startswith('%') or '&' in next_line):
                    break

                if next_line.startswith('🔹') or any(
                        k in next_line.upper() for k in ['DICE', 'IOU', 'ACC', 'PRE', 'REC']):
                    body += next_line + "\n"
                    j += 1
                else:
                    break

            model = clean_model_name(header.replace('🏆', ''))

            try:
                dice = float(re.search(r'Dice.*?([\d\.]+)', body, re.IGNORECASE).group(1))
                iou = float(re.search(r'IoU.*?([\d\.]+)', body, re.IGNORECASE).group(1))
                acc = float(re.search(r'ACC.*?([\d\.]+)', body, re.IGNORECASE).group(1))
                pre = float(re.search(r'Pre.*?([\d\.]+)', body, re.IGNORECASE).group(1))
                rec = float(re.search(r'Rec.*?([\d\.]+)', body, re.IGNORECASE).group(1))

                metrics = [dice, iou, acc, pre, rec]
                if model:
                    # 最新终端日志直接覆盖旧数据
                    datasets[current_ds][model] = metrics
            except (AttributeError, ValueError):
                pass

            i = j - 1

        i += 1

    # === 输出并生成最终格式 ===
    for ds_name, ds_data in datasets.items():
        if not ds_data: continue

        if "DPRASR-Net" not in ds_data:
            ds_data["DPRASR-Net"] = [0.0, 0.0, 0.0, 0.0, 0.0]

        sorted_models = sorted([m for m in ds_data.items() if m[0] != "DPRASR-Net"], key=lambda x: x[1][0])
        sorted_models.append(("DPRASR-Net", ds_data["DPRASR-Net"]))

        max_v = []
        sub_v = []
        for col in range(5):
            vals = sorted(list(set([m[1][col] for m in sorted_models if m[1][col] > 0])), reverse=True)
            max_v.append(vals[0] if len(vals) > 0 else 0)
            sub_v.append(vals[1] if len(vals) > 1 else (vals[0] if len(vals) > 0 else 0))

        print(f"\n% === {ds_name} Dataset Table ===")
        for model, metrics in sorted_models:
            row_str = []
            row_name = "\\textbf{DPRASR-Net (Ours)}" if model == "DPRASR-Net" else f"{model} {cites.get(model, '')}".strip()

            if metrics[0] == 0:
                row_str = ["-", "-", "-", "-", "-"]
            else:
                for c in range(5):
                    val = metrics[c]
                    if val == max_v[c]:
                        row_str.append(f"\\textbf{{{val:.4f}}}")
                    elif val == sub_v[c]:
                        row_str.append(f"\\underline{{{val:.4f}}}")
                    else:
                        row_str.append(f"{val:.4f}")

            if model == "DPRASR-Net":
                print("\\midrule")

            print(f"{row_name} & " + " & ".join(row_str) + " \\\\")


if __name__ == "__main__":
    main()