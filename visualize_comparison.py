# -*- coding: utf-8 -*-
"""visualize_comparison.py
============================================================
Generate segmentation black and white mask comparison images of multiple models on 2 data sets.
Layout: 4 rows × 10 columns (number of dynamically adapted models)
  - 4 rows: 2 ultrasound images from each of the 2 data sets (BUSI/UDIAT)
  - 10 columns: original image | GT | DPRASRNet (green title, ours) | 7 comparison models
        (UNet / DBF-Net / HiFormer / MISSFormer / MK-UNet / CMU-Net / UNet++)

Usage:
    python visualize_comparison.py
    python visualize_comparison.py --n 3 # Take 3 images from each data set (then 6 rows)
    python visualize_comparison.py --seed 100 # Change a group of random pictures
    python visualize_comparison.py --out cmp.png"""

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

# ---------- Safe import set_seed ----------
try:
    from seed.set_seed import set_seed
except ImportError:

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

            # ---------- Path ----------


DPRASR = Path(__file__).resolve().parent
DATA = (DPRASR / ".." / "data").resolve()

# Only add the root directory to sys.path to avoid files with the same name in subfolders from polluting the global namespace.
if str(DPRASR) not in sys.path:
    sys.path.insert(0, str(DPRASR))

    # segformer Download nvidia/mit-b0 from HuggingFace and use domestic mirroring
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


# ---------- Isolated import context manager ----------
@contextlib.contextmanager
def model_import_context(model_dir_path):
    """Temporarily switch working directories and isolate sys.path:
    1. Prevent utils.py / config.py conflicts between multiple models
    2. Ensure that the relative path inside the model (such as ./configs/) can be read correctly during import and instantiation.
    """
    old_cwd = os.getcwd()
    model_dir_str = str(model_dir_path)

    # Temporarily insert the model directory to the front of sys.path
    sys.path.insert(0, model_dir_str)
    if model_dir_path.exists():
        os.chdir(model_dir_str)

    try:
        yield
    finally:
        os.chdir(old_cwd)
        if model_dir_str in sys.path:
            sys.path.remove(model_dir_str)

            # ----------Dataset configuration ----------


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

# ---------- Model configuration (column order: DPRASRNet column 3, the remaining 7 comparisons) ----------
MODELS = [
    {"key": "DPRASRNet", "name": "DPRASR-Net", "ours": True, "dir": "DPRASRNet"},
    {"key": "unet", "name": "UNet", "ours": False, "dir": "unet"},
    {"key": "DBFNet", "name": "DBF-Net", "ours": False, "dir": "DBFNet"},
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
    """The input size of each model when training on each data set"""
    if model_key == "hiformer":
        return 224
    if model_key == "missformer":
        return 256 if dataset == "udiat" else 224
    return 256

    # ---------- Model loading (isolated import + directory error prevention) ----------


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

        elif model_key == "unet":
            with model_import_context(DPRASR / "unet"):
                try:
                    from main import UNet
                except ImportError:
                    from unet.main import UNet
                model = UNet(n_channels=3, n_classes=1)

        elif model_key == "DBFNet":
            with model_import_context(DPRASR / "DBFNet"):
                try:
                    from DBFNet.main import DBFNet
                except ImportError:
                    from main import DBFNet
                model = DBFNet(in_channels=3, num_classes=1)

        elif model_key == "hiformer":
            with model_import_context(DPRASR / "hiformer"):
                try:
                    from hiformer.models.HiFormer import HiFormer
                    from hiformer.configs.HiFormer_configs import get_hiformer_b_configs
                except ImportError:
                    from models.HiFormer import HiFormer
                    from configs.HiFormer_configs import get_hiformer_b_configs
                model = HiFormer(
                    config=get_hiformer_b_configs(),
                    img_size=224,
                    in_chans=3,
                    n_classes=1,
                )

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
                activation="sigmoid",
            )

        else:
            return None

        state = torch.load(str(weight_p), map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model

    except Exception as e:
        print(f"[WARNING] Loading{model_key }fail({weight_p .name }): {e }")
        return None

        # ---------- Preprocessing/Inference ----------


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Models whose training pipeline applies ImageNet normalization
NORM_MODELS = {"unet", "DBFNet"}


def preprocess(img_pil, size, model_key=None):
    img = img_pil.convert("RGB").resize((size, size))
    t = TF.to_tensor(img)
    if model_key in NORM_MODELS:
        t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.unsqueeze(0).to(DEVICE)


@torch.no_grad()
def predict(model, model_key, img_tensor):
    out = model(img_tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]

        # Return raw logits to a model that requires manual sigmoid
    if model_key in ("hiformer", "MKUnet", "CMUNet"):
        out = torch.sigmoid(out)

    return out

    # ---------- Select picture ----------


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
                        alt = (
                            f.replace(".png", ".PNG")
                            if ".png" in f
                            else f.replace(".PNG", ".png")
                        )
                        if (gd / alt).exists():
                            mp = gd / alt
                    if mp.exists():
                        pairs.append((od / f, mp))
    return pairs


def pick_pairs(ds_key, n=2, seed=42):
    pairs = collect_pairs(DATASETS[ds_key])
    if not pairs:
        print(f"[warn]{ds_key }No pictures found:{DATASETS [ds_key ]['root']}")
        return []
    random.Random(seed).shuffle(pairs)
    return pairs[:n]

    # ---------- Drawing ----------


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
                ax.set_title(
                    title,
                    color="green" if is_ours else "black",
                    fontsize=11,
                    fontweight="bold" if is_ours else "normal",
                )

    for r, row in enumerate(rows):
        img = row["img_pil"].convert("RGB").resize((256, 256))
        mask = row["mask_pil"].convert("L").resize((256, 256))

        # Column 1 original image
        axes[r][0].imshow(img)
        axes[r][0].set_ylabel(
            f"{row ['ds']}\n{row ['img_name'][:18 ]}",
            fontsize=8,
            rotation=0,
            labelpad=40,
            va="center",
        )
        # Column 2 GT
        axes[r][1].imshow(np.array(mask), cmap="gray")

        # The following columns model predictions
        for c, m in enumerate(MODELS, start=2):
            ax = axes[r][c]
            pred = row["preds"].get(m["key"])
            if pred is None:
                ax.imshow(np.zeros((256, 256)), cmap="gray", vmin=0, vmax=1)
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="red",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax.imshow(pred, cmap="gray", vmin=0, vmax=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ The comparison chart has been saved:{out_path }")

    # ---------- Main process ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n",
        type=int,
        default=2,
        help="How many pictures are taken from each data set (default 2)",
    )
    ap.add_argument(
        "--out", type=str, default="comparison.png", help="Output file name"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed, change the number to select different pictures",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"DEVICE: {DEVICE } | DPRASR: {DPRASR }")
    print(
        f"DATA: {DATA }| Taken from each data set{args .n }Zhang | Seed:{args .seed }| Output:{args .out }"
    )

    # 1. Select pictures
    all_rows = []
    for ds_key in ["busi", "udiat"]:
        pairs = pick_pairs(ds_key, n=args.n, seed=args.seed)
        for img_p, mask_p in pairs:
            all_rows.append(
                {
                    "ds": ds_key,
                    "img_name": Path(img_p).name,
                    "img_pil": Image.open(img_p),
                    "mask_pil": Image.open(mask_p),
                    "ds_key": ds_key,
                    "preds": {},
                }
            )
        print(f"  [{ds_key }] select{len (pairs )}open")

    if not all_rows:
        print("❌ No pictures were selected, check the data path.")
        return

        # 2. For each model: load weights, inference
    for m in MODELS:
        print(f"\n=== Model{m ['name']} ===")
        for ds_key in ["busi", "udiat"]:
            wp = weight_path(m["key"], ds_key)
            if not wp.exists():
                print(
                    f"  [{ds_key }] Weight missing:{wp .name }(This data set is skipped by this model, and N/A is drawn)"
                )
                continue
            print(f"  [{ds_key }] Load weights:{wp .name }")
            size = get_size(m["key"], ds_key)
            model = load_model(m["key"], wp, img_size=size)
            if model is None:
                continue

            for row in all_rows:
                if row["ds_key"] != ds_key:
                    continue
                img_t = preprocess(row["img_pil"], size, m["key"])
                pred = predict(model, m["key"], img_t)
                pred_np = pred.squeeze().cpu().numpy()
                pred_bin = (pred_np > 0.5).astype(np.uint8) * 255

                if pred_bin.shape != (256, 256):
                    pred_bin = np.array(Image.fromarray(pred_bin).resize((256, 256)))
                row["preds"][m["key"]] = pred_bin

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # 3. Draw a picture
    draw_grid(all_rows, args.out)


if __name__ == "__main__":
    main()
