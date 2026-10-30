"""
Ablation: DSADS dynamic deep-supervision weight vs. static weights vs. no DS.
=============================================================================

Trains the FULL model (DPRCAB + ACSSR + aux heads) repeatedly with everything
identical except the auxiliary deep-supervision weight:

  dynamic  -- calculate_ds_weight(train_size), the proposed DSADS (logistic)
  none     -- no deep supervision (w = 0)
  0.1 ...  -- fixed static weights, default sweep {0.1, 0.15, 0.2, 0.25, 0.3}

Run (from CLADS-Net root, with boundary_metrics.py in the same folder):
  python static_weight_ablation.py --dataset BUSI     --data_dir /path/to/Dataset_BUSI_with_GT
  python static_weight_ablation.py --dataset UDIAT    --data_dir /path/to/UDIAT_Dataset_B
  python static_weight_ablation.py --dataset BUS_UCLM --data_dir "/path/to/BUS-UCLM ..."
  python static_weight_ablation.py --dataset BUSI --data_dir ... --weights dynamic 0.1 0.2 0.3

Output: per-weight test metrics (region + HD95/ASSD/NSD) and a LaTeX-ready table.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boundary_metrics import boundary_metrics_batch


# ----------------------------------------------------------------
# 0. Seed & dynamic DS weight (identical to production main.py)
# ----------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_ds_weight(dataset_size, N_min=130.0, N_max=624.0,
                        W_min=0.15, W_max=0.30):
    mu = (N_min + N_max) / 2.0
    gamma = math.log(19) / (N_max - mu)
    return W_min + (W_max - W_min) / (1.0 + math.exp(-gamma * (dataset_size - mu)))


# ----------------------------------------------------------------
# 1. Full model (frozen copy of the production architecture; if your
#    local CLADS-Net classes were renamed/changed, replace CLADSNet
#    below with your local definition)
# ----------------------------------------------------------------
class DPRCAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // 4, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        feat = self.LConv(x)
        att = self.sigmoid(self.mlp(self.gap(feat)) + self.mlp(self.gmp(feat)))
        return feat * att + identity


class AdaptiveCrossScaleSoftRouting(nn.Module):
    """ACSSR: pixel-wise soft routing with a global (GAP) context term."""

    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid_channels = max(1, (channels * num_scales) // reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels * num_scales, kernel_size=1, bias=False),
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        concat_f = torch.cat(features, dim=1)
        attn_weights = self.router(concat_f)                          # (B, K*C, H, W)
        attn_weights = attn_weights + self.router(self.gap(concat_f)) # + GAP context term
        attn_weights = attn_weights.view(B, self.num_scales, C, H, W)
        attn_weights = F.softmax(attn_weights, dim=1)
        stacked_f = torch.stack(features, dim=1)
        fused_out = (stacked_f * attn_weights).sum(dim=1)
        return fused_out


class ChannelAdapter(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.transpose(1, 2).reshape(B, -1, H, W)


class CLADSNet(nn.Module):
    """Full model, identical to production DPRASR_Net / CLADS-Net."""

    def __init__(self, embed_dim=256):
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

        self.acssr = AdaptiveCrossScaleSoftRouting(embed_dim, num_scales=5, reduction=4)

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

        fused = self.acssr(l0, l1, l2, l3, lb)
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
# 2. Loss & metrics (identical to production)
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
        p = pred.view(pred.size(0), -1)
        g = target.view(target.size(0), -1)
        inter = (p * g).sum(dim=1)
        dice = (2.0 * inter + smooth) / (p.sum(1) + g.sum(1) + smooth)
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
# 3. Datasets (BUSI / UDIAT / BUS-UCLM, same transforms as production)
# ----------------------------------------------------------------
class MedicalDataset(Dataset):
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


def get_paths(data_dir, dataset):
    img_p, mask_p = [], []
    if dataset == "BUSI":  # {benign,malignant}/*.png + *_mask.png
        for cat in ["benign", "malignant"]:
            cat_dir = os.path.join(data_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for f in os.listdir(cat_dir):  # plain listdir: mirrors production split
                if f.endswith(".png") and not f.endswith("_mask.png"):
                    mp = os.path.join(cat_dir, f.replace(".png", "_mask.png"))
                    if os.path.exists(mp):
                        img_p.append(os.path.join(cat_dir, f))
                        mask_p.append(mp)
    elif dataset == "BUS_UCLM":  # {benign,malign}/*.png + flat masks/<name>.png
        masks_dir = os.path.join(data_dir, "masks")
        for cat in ["benign", "malign"]:
            cat_dir = os.path.join(data_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            for f in os.listdir(cat_dir):  # plain listdir: mirrors production split
                if f.endswith(".png"):
                    mp = os.path.join(masks_dir, f)
                    if os.path.exists(mp):
                        img_p.append(os.path.join(cat_dir, f))
                        mask_p.append(mp)
    else:  # UDIAT: original/ + GT/
        img_dir, gt_dir = os.path.join(data_dir, "original"), os.path.join(data_dir, "GT")
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
# 4. Train + evaluate one DS-weight configuration
# ----------------------------------------------------------------
def train_and_eval(weight_tag, ds_w, loaders, train_size, device, args):
    set_seed(args.seed)
    model = CLADSNet().to(device)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

    if weight_tag == "dynamic":
        ds_w = calculate_ds_weight(train_size)
    print(f"  [DS={weight_tag}] aux weight = {ds_w:.4f}")

    train_loader, val_loader, test_loader = loaders
    save_path = os.path.join(args.out_dir, f"best_{args.dataset}_ds{weight_tag}_seed{args.seed}.pth")
    best_val, patience = 0.0, (5 if args.dataset == "BUSI" else 10)
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"[DS={weight_tag}] ep{epoch + 1}/{args.epochs}",
                                  leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            out, o0, o1, o2, o3 = model(images)
            loss = criterion(out, masks)
            if ds_w > 0.0:
                loss = loss + ds_w * (criterion(o0, masks) + criterion(o1, masks)
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

        if np.mean(val_dices) > best_val:
            best_val = float(np.mean(val_dices))
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return evaluate(model, test_loader, device, args.nsd_tau)


@torch.no_grad()
def evaluate(model, loader, device, nsd_tau=2.0):
    model.eval()
    region = {k: [] for k in ["dice", "iou", "acc", "pre", "rec"]}
    bnd = {"hd95": [], "assd": [], "nsd": []}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        out = model(images)
        for k in region:
            region[k].extend(calculate_region_metrics(out, masks)[k])
        res = boundary_metrics_batch(out, masks, tau=nsd_tau)
        for k in bnd:
            if not math.isnan(res[k]):
                bnd[k].append(res[k])
    r = {"Dice": float(np.mean(region["dice"])), "IoU": float(np.mean(region["iou"])),
         "ACC": float(np.mean(region["acc"])), "Pre": float(np.mean(region["pre"])),
         "Rec": float(np.mean(region["rec"]))}
    r["HD95"] = float(np.mean(bnd["hd95"])) if bnd["hd95"] else float("nan")
    r["ASSD"] = float(np.mean(bnd["assd"])) if bnd["assd"] else float("nan")
    r["NSD"] = float(np.mean(bnd["nsd"])) if bnd["nsd"] else float("nan")
    return r


# ----------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["BUSI", "UDIAT", "BUS_UCLM"], default="BUSI")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--weights", nargs="+",
                    default=["dynamic", "none", "0.1", "0.15", "0.2", "0.25", "0.3"],
                    help="tokens: 'dynamic', 'none', or numeric static weights")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--nsd_tau", type=float, default=2.0)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | dataset: {args.dataset} | seed: {args.seed}")

    all_imgs, all_masks = get_paths(args.data_dir, args.dataset)
    print(f"Found {len(all_imgs)} image-mask pairs in {args.data_dir}")
    assert len(all_imgs) > 0, "no data found, check --data_dir"

    set_seed(args.seed)
    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    n = len(combined)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    splits = (combined[:n_train], combined[n_train:n_train + n_val], combined[n_train + n_val:])
    loaders = tuple(
        DataLoader(MedicalDataset([p for p, _ in s], [m for _, m in s], is_train=(i == 0)),
                   batch_size=args.batch_size, shuffle=(i == 0))
        for i, s in enumerate(splits))

    results = {}
    for tag in args.weights:
        if tag not in ("dynamic", "none"):
            try:
                float(tag)
            except ValueError:
                raise ValueError(f"invalid weight token: {tag!r}")
        print(f"\n=== DS weight: {tag} ===")
        results[tag] = train_and_eval(tag, 0.0, loaders, n_train, device, args)
        r = results[tag]
        print(f"  -> DS={tag}: Dice={r['Dice']:.4f} IoU={r['IoU']:.4f} "
              f"HD95={r['HD95']:.2f} ASSD={r['ASSD']:.2f} NSD={r['NSD']:.4f}")

    keys = ["Dice", "IoU", "ACC", "Pre", "Rec", "HD95", "ASSD", "NSD"]
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Deep-supervision weighting ablation on " + args.dataset + r": dynamic "
        r"(dataset-size-aware logistic) vs. static weights vs. no deep supervision "
        r"(identical split, protocol, and seed " + str(args.seed) + r").}",
        r"\label{tab:ds_weight_ablation}",
        r"\begin{tabular}{l|ccccc|ccc}", r"\hline",
        r"DS weight & Dice & IoU & ACC & Pre & Rec & HD95$\downarrow$ & ASSD$\downarrow$ & NSD$\uparrow$ \\",
        r"\hline",
    ]

    def disp(tag):
        if tag == "dynamic":
            return r"Dynamic, dataset-size-aware (proposed)"
        if tag == "none":
            return r"w/o deep supervision"
        return rf"Static $w={tag}$"

    for tag, r in results.items():
        cells = [disp(tag)]
        for k in keys:
            txt = rf"{r[k]:.4f}" if k not in ("HD95", "ASSD") else rf"{r[k]:.2f}"
            cells.append(txt)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]

    out_txt = os.path.join(args.out_dir, f"ds_weight_ablation_results_{args.dataset}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\nLaTeX table saved to", out_txt)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
