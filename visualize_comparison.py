# -*- coding: utf-8 -*-
"""
visualize_comparison.py
============================================================
生成多个模型在 2 个数据集上的分割黑白掩膜对比图。
布局：4 行 × 11 列 (动态适配模型数量)
  - 4 行：2 个数据集(BUSI/UDIAT)各取 2 张超声图
  - 11 列：原图 | GT | DPRASRNet(绿色标题,我们的) | 8 个对比模型
        (SegFormer / Swin-UNet / TransUNet / HiFormer / MISSFormer / MK-UNet / CMU-Net / UNet++)

用法：
    python visualize_comparison.py
    python visualize_comparison.py --n 3      # 每数据集取3张(则6行)
    python visualize_comparison.py --seed 100 # 换一组随机图片
    python visualize_comparison.py --out cmp.png
"""
import os
import sys
import random
import argparse
import contextlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 安全导入 set_seed ----------
try:
    from seed.set_seed import set_seed
except ImportError:
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

# ---------- 路径 ----------
DPRASR = Path(__file__).resolve().parent
DATA = (DPRASR / ".." / "data").resolve()

# 只将根目录加入 sys.path，避免子文件夹同名文件污染全局命名空间
if str(DPRASR) not in sys.path:
    sys.path.insert(0, str(DPRASR))

# segformer 从 HuggingFace 下载 nvidia/mit-b0，走国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ---------- 隔离导入上下文管理器 ----------
@contextlib.contextmanager
def model_import_context(model_dir_path):
    """
    临时切换工作目录并隔离 sys.path：
    1. 防止多模型之间的 utils.py / config.py 相互冲突
    2. 确保在 import 与 实例化 时，模型内部的相对路径（如 ./configs/）能够正确读取
    """
    old_cwd = os.getcwd()
    model_dir_str = str(model_dir_path)

    # 临时把模型目录插到 sys.path 最前面
    sys.path.insert(0, model_dir_str)
    if model_dir_path.exists():
        os.chdir(model_dir_str)

    try:
        yield
    finally:
        os.chdir(old_cwd)
        if model_dir_str in sys.path:
            sys.path.remove(model_dir_str)


# ---------- 数据集配置 ----------
DATASETS = {
    "busi": {
        "root": DATA / "Dataset_BUSI_with_GT",
        "kind": "busi",
    },
    "udiat": {
        "root": DATA / "UDIAT_Dataset_B",
        "kind": "udiat",
    },
}

# ---------- 模型配置（列顺序：DPRASRNet 第3列，其余8个对比）----------
MODELS = [
    {"key": "DPRASRNet", "name": "DPRASR-Net", "ours": True, "dir": "DPRASRNet"},
    {"key": "segformer", "name": "SegFormer", "ours": False, "dir": "segformer"},
    {"key": "swinunet", "name": "Swin-UNet", "ours": False, "dir": "swinunet"},
    {"key": "transunet", "name": "TransUNet", "ours": False, "dir": "transunet"},
    {"key": "hiformer", "name": "HiFormer", "ours": False, "dir": "hiformer"},
    {"key": "missformer", "name": "MISSFormer", "ours": False, "dir": "missformer"},
    {"key": "MKUnet", "name": "MK-UNet", "ours": False, "dir": "MKUnet"},
    {"key": "CMUNet", "name": "CMU-Net", "ours": False, "dir": "CMUNet"},
    {"key": "unetplusplus", "name": "UNet++", "ours": False, "dir": "unetplusplus"},
]

WEIGHT_NAMES = {
    "busi": "best_busi.pth",
    "udiat": "best_UDIAT.pth",
}


def weight_path(model_key, dataset):
    name = WEIGHT_NAMES[dataset]
    model_dir = MODELS[[m["key"] for m in MODELS].index(model_key)]["dir"]

    p = DPRASR / model_dir / name
    if not p.exists() and model_key == "DPRASRNet":
        p = DPRASR / "DPRASRNet" / name
    return p


def get_size(model_key, dataset):
    """各模型在各数据集训练时的输入尺寸"""
    if model_key in ("swinunet", "hiformer"):
        return 224
    if model_key == "missformer":
        return 256 if dataset == "udiat" else 224
    return 256


# ---------- 模型加载（隔离导入 + 目录防错） ----------
def load_model(model_key, weight_p, img_size=None):
    try:
        if model_key == "DPRASRNet":
            DPRASR_dir = DPRASR / "DPRASRNet"
            with model_import_context(DPRASR_dir):
                try:
                    from main import DPRASR_Net
                    model = DPRASR_Net(embed_dim=256)
                except ImportError:
                    from main import DPRASR_Net
                    model = DPRASR_Net(embed_dim=256)

        elif model_key == "segformer":
            with model_import_context(DPRASR / "segformer"):
                try:
                    from segformer.main import SegFormerWrapper
                except ImportError:
                    from main import SegFormerWrapper
                model = SegFormerWrapper(pretrained_model="nvidia/mit-b0", num_classes=1)

        elif model_key == "swinunet":
            with model_import_context(DPRASR / "swinunet"):
                # 注意：必须在 chdir(swinunet) 之后 import，否则其内部读取 ./configs/ 会找不到文件
                try:
                    from swinunet.main import OfficialSwinUNetWrapper
                except ImportError:
                    from main import OfficialSwinUNetWrapper
                model = OfficialSwinUNetWrapper(img_size=224, num_classes=1)

        elif model_key == "transunet":
            with model_import_context(DPRASR / "transunet"):
                try:
                    from transunet.main import TransUNetWrapper
                    model = TransUNetWrapper(img_size=img_size or 224, num_classes=1)
                except ImportError:
                    try:
                        from networks.vit_seg_modeling import VisionTransformer as ViT_seg
                        from networks.vit_seg_modeling import CONFIGS
                        config_vit = CONFIGS['R50-ViT-B_16']
                        config_vit.n_classes = 1
                        config_vit.n_skip = 3
                        if img_size:
                            config_vit.patches.grid = (int(img_size / 16), int(img_size / 16))
                        model = ViT_seg(config_vit, img_size=img_size or 224, num_classes=1)
                    except ImportError:
                        from main import TransUNetWrapper
                        model = TransUNetWrapper(img_size=img_size or 224, num_classes=1)

        elif model_key == "hiformer":
            with model_import_context(DPRASR / "hiformer"):
                try:
                    from hiformer.models.HiFormer import HiFormer
                    from hiformer.configs.HiFormer_configs import get_hiformer_b_configs
                except ImportError:
                    from models.HiFormer import HiFormer
                    from configs.HiFormer_configs import get_hiformer_b_configs
                model = HiFormer(config=get_hiformer_b_configs(), img_size=224, in_chans=3, n_classes=1)

        elif model_key == "missformer":
            with model_import_context(DPRASR / "missformer"):
                try:
                    from missformer.main import MISSFormerWrapper
                except ImportError:
                    from main import MISSFormerWrapper
                model = MISSFormerWrapper(n_classes=1, img_size=img_size or 224)

        elif model_key == "MKUnet":
            with model_import_context(DPRASR / "MKUnet"):
                try:
                    from mkunet_network import MK_UNet
                except ImportError:
                    from main import MK_UNet
                model = MK_UNet(num_classes=1, in_channels=3)

        elif model_key == "CMUNet":
            with model_import_context(DPRASR / "CMUNet"):
                from src.network.CMUNet import CMUNet
                model = CMUNet(img_ch=3, output_ch=1, l=7, k=7)

        elif model_key == "unetplusplus":
            import segmentation_models_pytorch as smp
            model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                activation="sigmoid"
            )

        else:
            return None

        state = torch.load(str(weight_p), map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model

    except Exception as e:
        print(f"  [警告] 加载 {model_key} 失败({weight_p.name}): {e}")
        return None


# ---------- 预处理 / 推理 ----------
def preprocess(img_pil, size):
    img = img_pil.convert("RGB").resize((size, size))
    t = TF.to_tensor(img)
    return t.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(model, model_key, img_tensor):
    out = model(img_tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]

    # 返回 raw logits 需手动 sigmoid 的模型
    if model_key in ("hiformer", "MKUnet", "CMUNet", "transunet"):
        out = torch.sigmoid(out)

    return out


# ---------- 选图 ----------
def collect_pairs(ds_cfg):
    root = ds_cfg["root"]
    kind = ds_cfg["kind"]
    pairs = []
    if kind == "busi":
        for cat in ["benign", "malignant"]:
            d = root / cat
            if not d.exists():
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".png") and not f.endswith("_mask.png"):
                    mp = d / f.replace(".png", "_mask.png")
                    if mp.exists():
                        pairs.append((d / f, mp))
    elif kind == "udiat":
        od, gd = root / "original", root / "GT"
        if od.exists() and gd.exists():
            for f in sorted(os.listdir(od)):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    mp = gd / f
                    if not mp.exists():
                        alt = f.replace(".png", ".PNG") if ".png" in f else f.replace(".PNG", ".png")
                        if (gd / alt).exists():
                            mp = gd / alt
                    if mp.exists():
                        pairs.append((od / f, mp))
    return pairs


def pick_pairs(ds_key, n=2, seed=42):
    pairs = collect_pairs(DATASETS[ds_key])
    if not pairs:
        print(f"[警告] {ds_key} 未找到图片: {DATASETS[ds_key]['root']}")
        return []
    random.Random(seed).shuffle(pairs)
    return pairs[:n]


# ---------- 画图 ----------
def draw_grid(rows, out_path):
    n_rows = len(rows)
    n_cols = 2 + len(MODELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ["Image", "GT"] + [m["name"] for m in MODELS]
    for c, title in enumerate(col_titles):
        is_ours = c >= 2 and MODELS[c - 2]["ours"]
        for r in range(n_rows):
            ax = axes[r][c]
            ax.axis("off")
            if r == 0:
                ax.set_title(title, color="green" if is_ours else "black",
                             fontsize=11, fontweight="bold" if is_ours else "normal")

    for r, row in enumerate(rows):
        img = row["img_pil"].convert("RGB").resize((256, 256))
        mask = row["mask_pil"].convert("L").resize((256, 256))

        # 第1列 原图
        axes[r][0].imshow(img)
        axes[r][0].set_ylabel(f"{row['ds']}\n{row['img_name'][:18]}",
                              fontsize=8, rotation=0, labelpad=40, va="center")
        # 第2列 GT
        axes[r][1].imshow(np.array(mask), cmap="gray")

        # 后面列 模型预测
        for c, m in enumerate(MODELS, start=2):
            ax = axes[r][c]
            pred = row["preds"].get(m["key"])
            if pred is None:
                ax.imshow(np.zeros((256, 256)), cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=12, fontweight="bold")
            else:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ 对比图已保存: {out_path}")


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="每个数据集取几张图(默认2)")
    ap.add_argument("--out", type=str, default="comparison.png", help="输出文件名")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，改变数字可挑选不同图片")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"DEVICE: {DEVICE} | DPRASR: {DPRASR}")
    print(f"DATA: {DATA} | 每数据集取 {args.n} 张 | 种子: {args.seed} | 输出: {args.out}")

    # 1. 选图
    all_rows = []
    for ds_key in ["busi", "udiat"]:
        pairs = pick_pairs(ds_key, n=args.n, seed=args.seed)
        for img_p, mask_p in pairs:
            all_rows.append({
                "ds": ds_key,
                "img_name": Path(img_p).name,
                "img_pil": Image.open(img_p),
                "mask_pil": Image.open(mask_p),
                "ds_key": ds_key,
                "preds": {},
            })
        print(f"  [{ds_key}] 选 {len(pairs)} 张")

    if not all_rows:
        print("❌ 未选到任何图片，检查数据路径。")
        return

    # 2. 对每个模型：加载权重，推理
    for m in MODELS:
        print(f"\n=== 模型 {m['name']} ===")
        for ds_key in ["busi", "udiat"]:
            wp = weight_path(m["key"], ds_key)
            if not wp.exists():
                print(f"  [{ds_key}] 权重缺失: {wp.name}（该数据集此模型跳过，画 N/A）")
                continue
            print(f"  [{ds_key}] 加载权重: {wp.name}")
            size = get_size(m["key"], ds_key)
            model = load_model(m["key"], wp, img_size=size)
            if model is None:
                continue

            for row in all_rows:
                if row["ds_key"] != ds_key:
                    continue
                img_t = preprocess(row["img_pil"], size)
                pred = predict(model, m["key"], img_t)
                pred_np = pred.squeeze().cpu().numpy()
                pred_bin = (pred_np > 0.5).astype(np.uint8) * 255

                if pred_bin.shape != (256, 256):
                    pred_bin = np.array(Image.fromarray(pred_bin).resize((256, 256)))
                row["preds"][m["key"]] = pred_bin

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. 画图
    draw_grid(all_rows, args.out)


if __name__ == "__main__":
    main()