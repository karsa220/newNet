"""
Controlled ablation: ACSSR vs SKNet-original-select vs pixel-wise ACSSR.
=========================================================================

Answers Reviewer 4 directly: "ACSSR is SKNet's select step applied to K scales;
the only change is concatenating per-branch pooled descriptors instead of
summing branches first." This script isolates exactly that change.

Design (all confounds controlled):
  * identical backbone (DenseNet-121, ImageNet init), DPRCAB positions,
    ChannelAdapter embeddings, aux heads, loss, optimizer, LR schedule,
    early stopping, data split, and augmentation across variants;
  * ONLY the 5-scale fusion module is swapped (see fusion_variants.py):
      V1 sknet_select   -- SKNet's original select (sum -> GAP -> FCs -> softmax)
      V2 acssr_channel  -- paper Eqs. (5)-(8) (per-scale GAP -> concat -> MLP)
      V3 acssr_pixel    -- released-code version (pixel-wise routing)
  * region metrics (Dice/IoU/ACC/Pre/Rec) + boundary metrics (HD95/ASSD/NSD)
  * multi-seed support -> mean +/- std per variant;
  * reports routing-module parameter counts;
  * outputs a LaTeX-ready table.

Run (from repo root, PYTHONPATH includes repo root):
  python DPRASRNet/acssr_ablation.py --dataset BUSI --data_dir /path/to/Dataset_BUSI_with_GT \
      --seeds 42 43 44
  python DPRASRNet/acssr_ablation.py --dataset UDIAT --data_dir /path/to/UDIAT_Dataset_B \
      --seeds 42 43 44

Dependencies: torch, torchvision, numpy, tqdm, scipy.
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seed.set_seed import set_seed
from DPRASRNet.fusion_variants import VARIANTS, build_fusion, count_params
from DPRASRNet.boundary_metrics import boundary_metrics_batch

from main import DPRCAB, ChannelAdapter, AdaptiveCrossScaleSoftRouting


# ----------------------------------------------------------------
# 1. Deep-supervision weight (identical to main.py)
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# 2. Shared building blocks (identical to main.py)
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# 3. Fusion-ablation network: only the fusion module is swapped
# ----------------------------------------------------------------
class FusionAblationNet(nn.Module):
    def __init__(self, embed_dim=256, fusion="acssr_channel"):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features
        self.ch0, self.ch2, self.ch3, self.ch4, self.ch_b = 64, 256, 512, 1024, 1024

        self.rcab1 = DPRCAB(self.ch2)
        self.rcab2 = DPRCAB(self.ch3)
        self.rcab3 = DPRCAB(self.ch4)

        self.linear0 = ChannelAdapter(self.ch0, embed_dim)
        self.linear1 = ChannelAdapter(self.ch2, embed_dim)
        self.linear2 = ChannelAdapter(self.ch3, embed_dim)
        self.linear3 = ChannelAdapter(self.ch4, embed_dim)
        self.linear_b = ChannelAdapter(self.ch_b, embed_dim)

        self.fusion = build_fusion(fusion, channels=embed_dim, num_scales=5)

        self.final_conv = nn.Conv2d(embed_dim, 1, 1)
        self.aux0 = nn.Conv2d(embed_dim, 1, 1)
        self.aux1 = nn.Conv2d(embed_dim, 1, 1)
        self.aux2 = nn.Conv2d(embed_dim, 1, 1)
        self.aux3 = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x)))
        e1 = self.encoder.denseblock1(self.encoder.pool0(e0))
        e2 = self.encoder.denseblock2(self.encoder.transition1(e1))
        e3 = self.encoder.denseblock3(self.encoder.transition2(e2))
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e3)))

        s1, s2, s3 = self.rcab1(e1), self.rcab2(e2), self.rcab3(e3)
        target_size = e0.shape[2:]

        l0 = self.linear0(e0)
        l1 = F.interpolate(self.linear1(s1), target_size, mode="bilinear", align_corners=False)
        l2 = F.interpolate(self.linear2(s2), target_size, mode="bilinear", align_corners=False)
        l3 = F.interpolate(self.linear3(s3), target_size, mode="bilinear", align_corners=False)
        lb = F.interpolate(self.linear_b(b), target_size, mode="bilinear", align_corners=False)

        fused = self.fusion(l0, l1, l2, l3, lb)

        out = torch.sigmoid(F.interpolate(self.final_conv(fused), size=x.shape[2:],
                                          mode="bilinear", align_corners=False))
        if self.training:
            aux = [self.aux0(l0), self.aux1(l1), self.aux2(lb), self.aux3(l3)]
            aux = [torch.sigmoid(F.interpolate(a, size=x.shape[2:],
                                               mode="bilinear", align_corners=False))
                   for a in aux]
            return (out, *aux)
        return out


# ----------------------------------------------------------------
# 4. Loss & region metrics (identical to main.py)
# ----------------------------------------------------------------
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        inter = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * inter + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        return self.weight_bce * bce_loss + self.weight_dice * (1.0 - dice.mean())


def calculate_region_metrics(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    target_bin = target.float()
    p = pred_bin.view(pred_bin.size(0), -1)
    g = target_bin.view(target_bin.size(0), -1)
    tp = (p * g).sum(dim=1)
    fp = (p * (1 - g)).sum(dim=1)
    fn = ((1 - p) * g).sum(dim=1)
    tn = ((1 - p) * (1 - g)).sum(dim=1)
    return {
        "dice": ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).tolist(),
        "iou": ((tp + smooth) / (tp + fp + fn + smooth)).tolist(),
        "acc": ((tp + tn + smooth) / (tp + fp + fn + tn + smooth)).tolist(),
        "pre": ((tp + smooth) / (tp + fp + smooth)).tolist(),
        "rec": ((tp + smooth) / (tp + fn + smooth)).tolist(),
    }


# ----------------------------------------------------------------
# 5. Datasets (re-implemented locally, identical transforms)
# ----------------------------------------------------------------
class BUSDataset(Dataset):
    """Covers BUSI (benign/malignant folders with *_mask.png) and UDIAT
    (original/ + GT/ folders), selected by layout detection."""

    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))
        if self.is_train:
            if random.random() > 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image, mask = TF.rotate(image, angle), TF.rotate(mask, angle)
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)
        return TF.to_tensor(image), TF.to_tensor(mask)


def _resolve_dir(data_dir, names):
    """Find a directory at data_dir or up to two levels below that contains
    every entry in `names` (case-insensitive). Returns (dir, child_map)."""
    names = [n.lower() for n in names]

    def child_map(d):
        return {e.lower(): e for e in os.listdir(d)
                if os.path.isdir(os.path.join(d, e))}

    frontier = [(data_dir, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        cm = child_map(d)
        if all(n in cm for n in names):
            return d, cm
        if depth < 2:
            frontier.extend((os.path.join(d, v), depth + 1) for v in cm.values())
    return None, {}


def get_paths(data_dir, dataset):
    img_p, mask_p = [], []
    if dataset == "BUSI":  # Dataset_BUSI_with_GT/{benign,malignant}/*.png (+_mask.png)
        base, cm = _resolve_dir(data_dir, ["benign", "malignant"])
        if base is None:
            base = data_dir
            cm = {e.lower(): e for e in os.listdir(data_dir)}
        print(f"  data root: {base}")
        for cat in ["benign", "malignant"]:
            cat_dir = os.path.join(base, cm.get(cat, cat))
            if not os.path.isdir(cat_dir):
                continue
            for f in sorted(os.listdir(cat_dir)):
                if f.endswith(".png") and not f.endswith("_mask.png"):
                    mp = os.path.join(cat_dir, f.replace(".png", "_mask.png"))
                    if os.path.exists(mp):
                        img_p.append(os.path.join(cat_dir, f))
                        mask_p.append(mp)
    elif dataset == "BUS_UCLM":  # {benign,malign}/*.png + masks/*.png (same filenames)
        base, cm = _resolve_dir(data_dir, ["benign", "masks"])
        if base is None:
            base = data_dir
            cm = {e.lower(): e for e in os.listdir(data_dir)}
        print(f"  data root: {base}")
        masks_dir = os.path.join(base, cm.get("masks", "masks"))
        for cat in ["benign", "malign"]:
            cat_dir = os.path.join(base, cm.get(cat, cat))
            if not os.path.isdir(cat_dir):
                continue
            for f in sorted(os.listdir(cat_dir)):
                if f.endswith(".png"):
                    mp = os.path.join(masks_dir, f)
                    if os.path.exists(mp):
                        img_p.append(os.path.join(cat_dir, f))
                        mask_p.append(mp)
    else:  # UDIAT: original/ + GT/  (or images/ + masks/)
        base, cm = _resolve_dir(data_dir, ["original", "gt"])
        if base is not None:
            img_key, gt_key = "original", "gt"
        else:
            base, cm = _resolve_dir(data_dir, ["images", "masks"])
            if base is None:
                raise FileNotFoundError(
                    f"neither original/GT nor images/masks found under {data_dir}; "
                    f"top-level entries: {os.listdir(data_dir)}")
            img_key, gt_key = "images", "masks"
        print(f"  data root: {base}")
        img_dir = os.path.join(base, cm[img_key])
        gt_dir = os.path.join(base, cm[gt_key])
        for f in sorted(os.listdir(img_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                mp = os.path.join(gt_dir, f)
                if not os.path.exists(mp):
                    alt = f.replace(".PNG", ".png") if ".PNG" in f else f.replace(".png", ".PNG")
                    mp = os.path.join(gt_dir, alt)
                if os.path.exists(mp):
                    img_p.append(os.path.join(img_dir, f))
                    mask_p.append(mp)
    return img_p, mask_p


# ----------------------------------------------------------------
# 6. Train + evaluate one (fusion, seed) pair
# ----------------------------------------------------------------
def train_and_eval(fusion, seed, loaders, device, train_size, args):
    set_seed(seed)
    ds_weight = 0.1
    model = FusionAblationNet(embed_dim=256, fusion=fusion).to(device)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

    train_loader, val_loader, test_loader = loaders
    save_path = os.path.join(args.out_dir, f"best_{args.dataset}_{fusion}_seed{seed}.pth")
    best_val_dice, patience, no_improve = 0.0, 10, 0

    if args.reuse and os.path.exists(save_path):
        print(f"  [reuse] {os.path.basename(save_path)} found -> skip training")
    else:
        for epoch in range(args.epochs):
            model.train()
            for images, masks in tqdm(train_loader,
                                      desc=f"[{fusion}|seed{seed}] ep{epoch + 1}/{args.epochs}",
                                      leave=False):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                out, o0, o1, o2, o3 = model(images)
                loss = criterion(out, masks) + ds_weight * (
                    criterion(o0, masks) + criterion(o1, masks)
                    + criterion(o2, masks) + criterion(o3, masks))
                loss.backward()
                optimizer.step()

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = model(images.to(device))
                    val_dices.extend(calculate_region_metrics(out, masks.to(device))["dice"])
            scheduler.step()

            if np.mean(val_dices) > best_val_dice:
                best_val_dice = np.mean(val_dices)
                torch.save(model.state_dict(), save_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # ----- test -----
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    region = {k: [] for k in ["dice", "iou", "acc", "pre", "rec"]}
    bnd = {"hd95": [], "assd": [], "nsd": []}
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            for k in region:
                region[k].extend(calculate_region_metrics(out, masks)[k])
            res = boundary_metrics_batch(out, masks, tau=args.nsd_tau)
            for k in bnd:
                if not math.isnan(res[k]):
                    bnd[k].append(res[k])

    r = {k.capitalize() if k != "acc" else "ACC": float(np.mean(v)) for k, v in region.items()}
    r.update({"HD95": float(np.mean(bnd["hd95"])) if bnd["hd95"] else float("nan"),
              "ASSD": float(np.mean(bnd["assd"])) if bnd["assd"] else float("nan"),
              "NSD": float(np.mean(bnd["nsd"])) if bnd["nsd"] else float("nan")})
    print(f"  -> [{fusion}|seed{seed}] Dice={r['Dice']:.4f} IoU={r['Iou']:.4f} "
          f"HD95={r['HD95']:.2f} ASSD={r['ASSD']:.2f} NSD={r['NSD']:.4f}")
    return r


# ----------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["BUSI", "UDIAT", "BUS_UCLM"], default="BUSI")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--nsd_tau", type=float, default=2.0,
                    help="NSD tolerance in pixels (evaluated at 256x256)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS.keys()),
                    default=list(VARIANTS.keys()),
                    help="routing variants to evaluate (subset of: "
                         + ", ".join(VARIANTS.keys()) + ")")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse best_<dataset>_<variant>_seed<seed>.pth in --out_dir "
                         "instead of training (test-only evaluation)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dataset: {args.dataset} | seeds: {args.seeds}")

    # parameter counts of the routing modules (the only differing part)
    print("\nRouting-module parameter counts (embed_dim=256, K=5):")
    for name in VARIANTS:
        m = build_fusion(name, channels=256, num_scales=5)
        print(f"  {name:<15}: {count_params(m):,}")

    all_imgs, all_masks = get_paths(args.data_dir, args.dataset)
    print(f"\nFound {len(all_imgs)} image-mask pairs in {args.data_dir}")
    assert len(all_imgs) > 0, "no data found, check --data_dir"

    results = {}  # fusion -> list of per-seed dicts
    for fusion in args.variants:
        results[fusion] = []
        for seed in args.seeds:
            print(f"\n=== variant: {fusion} | seed: {seed} ===")
            set_seed(seed)
            combined = list(zip(all_imgs, all_masks))
            random.shuffle(combined)
            n = len(combined)
            n_train = int((1 - args.val_ratio - args.test_ratio) * n)
            n_val = int(args.val_ratio * n)
            splits = (combined[:n_train],
                      combined[n_train:n_train + n_val],
                      combined[n_train + n_val:])
            loaders = tuple(
                DataLoader(BUSDataset([p for p, _ in s], [m for _, m in s],
                                      is_train=(i == 0)),
                           batch_size=args.batch_size, shuffle=(i == 0))
                for i, s in enumerate(splits))
            res = train_and_eval(fusion, seed, loaders, device, n_train, args)
            results[fusion].append(res)

    # ----- aggregate mean +/- std and emit LaTeX -----
    keys = ["Dice", "Iou", "ACC", "Pre", "Rec", "HD95", "ASSD", "NSD"]
    best = {k: max(np.mean([r[k] for r in v]) for v in results.values())
            for k in keys if k in ("Dice", "Iou", "NSD")}

    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Controlled comparison of cross-scale routing mechanisms "
        rf"(same backbone, DPRCAB, and training protocol; {args.dataset}; "
        rf"mean$\pm$std over {len(args.seeds)} seeds).}}",
        r"\label{tab:acssr_sknet}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l|ccccc|ccc}", r"\hline",
        r"Routing & Dice & IoU & ACC & Pre & Rec & HD95$\downarrow$ & ASSD$\downarrow$ & NSD$\uparrow$ \\",
        r"\hline",
    ]
    disp = {"sknet_select": r"SKNet select \cite{sknet}",
            "acssr_channel": r"ACSSR channel-wise (Eqs.~5--8)",
            "acssr_pixel": r"ACSSR pixel-wise (released code)",
            "acssr_pixel_channel": r"ACSSR pixel-wise + GAP (shared router)",
            "acssr_gap_only": r"ACSSR GAP-only (global routing)",
            "acssr_pixel_gap_sep": r"ACSSR pixel-wise + GAP (separate branch)"}
    for fusion, rows in results.items():
        cells = [disp.get(fusion, fusion)]
        for k in keys:
            vals = [r[k] for r in rows if not math.isnan(r[k])]
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            txt = rf"{mu:.4f}$\pm${sd:.4f}" if k in ("Dice", "Iou", "ACC", "Pre", "Rec", "NSD") \
                else rf"{mu:.2f}$\pm${sd:.2f}"
            if k in best and abs(mu - best[k]) < 1e-6:
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    out_txt = os.path.join(args.out_dir, "acssr_sknet_ablation_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "=" * 80)
    print("LaTeX table (saved to", out_txt + "):")
    print("=" * 80)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
