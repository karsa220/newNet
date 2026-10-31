import os
import random
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from main import AdaptiveCrossScaleSoftRouting, DPRCAB, ChannelAdapter, DPRASR_Net


# ==========================================
# 0. Random seed (to ensure the experiment is reproducible)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ==========================================
    # 1. Deep supervision weight calculation
    # ==========================================

    # ==========================================
    # 2. Network structure definition (DPRASR-Net)
    # ==========================================

        # ==========================================
        # 3. Loss function and indicator calculation
        # ==========================================


class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_score = (2.0 * intersection + smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
        )
        dice_loss = 1.0 - dice_score
        return (self.weight_bce * bce_loss) + (self.weight_dice * dice_loss.mean())


def calculate_metrics(pred, target, smooth=1e-5):
    pred_bin = (pred > 0.5).float()
    target_bin = target.float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target_bin.view(target_bin.size(0), -1)
    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1 - target_flat)).sum(dim=1)
    fn = ((1 - pred_flat) * target_flat).sum(dim=1)
    tn = ((1 - pred_flat) * (1 - target_flat)).sum(dim=1)

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    acc = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)
    pre = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    return {
        "dice": dice.tolist(),
        "iou": iou.tolist(),
        "acc": acc.tolist(),
        "pre": pre.tolist(),
        "rec": rec.tolist(),
    }

    # ==========================================
    # 4. Data set definition (BUSI & UDIAT unified processing)
    # ==========================================


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
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(
                    image, angle=0, translate=translate, scale=scale, shear=0
                )
                mask = TF.affine(
                    mask, angle=0, translate=translate, scale=scale, shear=0
                )

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


def get_busi_paths(data_dir):
    image_paths, mask_paths = [], []
    categories = ["benign", "malignant"]
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            continue
        for file_name in os.listdir(cat_dir):
            if file_name.endswith(".png") and not file_name.endswith("_mask.png"):
                img_path = os.path.join(cat_dir, file_name)
                mask_path = os.path.join(
                    cat_dir, file_name.replace(".png", "_mask.png")
                )
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
    return image_paths, mask_paths


def get_udiat_paths(data_dir):
    img_dir, gt_dir = os.path.join(data_dir, "original"), os.path.join(data_dir, "GT")
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"❌ Cannot find original or GT folder in{data_dir }")
        return [], []

    img_p, mask_p = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path, mask_path = os.path.join(img_dir, f), os.path.join(gt_dir, f)
            if os.path.exists(mask_path):
                img_p.append(img_path)
                mask_p.append(mask_path)
            else:
                alt_f = (
                    f.replace(".PNG", ".png")
                    if ".PNG" in f
                    else f.replace(".png", ".PNG")
                )
                alt_mask_path = os.path.join(gt_dir, alt_f)
                if os.path.exists(alt_mask_path):
                    img_p.append(img_path)
                    mask_p.append(alt_mask_path)
    return img_p, mask_p

    # ==========================================
    # 5. Main console
    # ==========================================


def main():
    set_seed(42)

    # 🌟🌟🌟 Control Panel 🌟🌟🌟
    MODE = "train"  # "train" training | "test" direct evaluation
    DATASET = "BUSI"  # "BUSI" | "UDIAT"

    # 🔥 Ablation Experiment Options 🔥
    # "static" : static weight (fixed to 0.2)
    # "none" : no depth supervision (weight is 0)
    SUPERVISION_MODE = "none"

    # Path configuration
    if DATASET == "BUSI":
        data_dir = r"../../data/Dataset_BUSI_with_GT"
        save_path = f"best_busi_{SUPERVISION_MODE }.pth"
        all_imgs, all_masks = get_busi_paths(data_dir)
    elif DATASET == "UDIAT":
        data_dir = r"../../data/UDIAT_Dataset_B"
        save_path = f"best_udiat_{SUPERVISION_MODE }.pth"
        all_imgs, all_masks = get_udiat_paths(data_dir)
    else:
        raise ValueError("Unsupported dataset selection!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"🚀 Device: {device } | 📐 Dataset: {DATASET } | 🔍 Mode: {SUPERVISION_MODE .upper ()} DS"
    )

    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ Dataset not found.")
        return

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # ------------------------------------------
    # Determine ds_weight based on ablation experiment configuration
    # ------------------------------------------
    if SUPERVISION_MODE == "static":
        ds_weight = 0.2
        print(f"🎯 DS Weight [Static]: {ds_weight :.4f}")
    elif SUPERVISION_MODE == "none":
        ds_weight = 0.0
        print(f"🎯 DS Weight [None]: {ds_weight :.4f} (Deep Supervision Disabled)")
    else:
        raise ValueError("SUPERVISION_MODE must be 'static' or 'none'")

    model = DPRASR_Net(embed_dim=256).to(device)

    # ==========================
    # Training process
    # ==========================
    if MODE == "train":
        train_loader = DataLoader(
            MedicalDataset(all_imgs[:train_size], all_masks[:train_size], True),
            batch_size=8,
            shuffle=True,
        )
        val_loader = DataLoader(
            MedicalDataset(
                all_imgs[train_size : train_size + val_size],
                all_masks[train_size : train_size + val_size],
                False,
            ),
            batch_size=8,
            shuffle=False,
        )

        criterion = HybridLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 10
        if DATASET == "BUSI":
            patience = 5

        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch +1 }/{num_epochs }")

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                out, out0, out1, out2, out3 = model(images)

                loss_main = criterion(out, masks)

                # 🔥 Melting core logic 🔥
                if SUPERVISION_MODE == "none":
                    loss = loss_main
                else:
                    loss_aux = ds_weight * (
                        criterion(out0, masks)
                        + criterion(out1, masks)
                        + criterion(out2, masks)
                        + criterion(out3, masks)
                    )
                    loss = loss_main + loss_aux

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({"Loss": f"{loss .item ():.4f}"})

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()
            print(
                f"📉 Epoch {epoch +1 } | Loss: {train_loss_epoch /len (train_loader ):.4f} | Val Dice: {avg_val_dice :.4f}"
            )

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Saved Best Model: {best_val_dice :.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stop trigger: continuous{patience }It is verified that Dice has not improved after 1 epoch and the training is stopped."
                    )
                    break

                    # ==========================
                    # Test process
                    # ==========================
    print("\n" + "=" * 50)
    print("🚀 Start evaluating on [Test Set]...")

    test_loader = DataLoader(
        MedicalDataset(
            all_imgs[train_size + val_size :], all_masks[train_size + val_size :], False
        ),
        batch_size=8,
        shuffle=False,
    )

    if not os.path.exists(save_path):
        print(
            f"❌ Weight file not found:{save_path }! Please run train mode training first."
        )
        return None

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

    # GPU warm-up
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_infer_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # Since the model only returns one out in eval() mode, there is no need to accept multiple return values.
            outputs = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_infer_time += end_time - start_time
            total_samples += images.size(0)

            metrics = calculate_metrics(outputs, masks)
            for k in test_res.keys():
                test_res[k].extend(metrics[k])

    avg_time_per_image = (total_infer_time / total_samples) * 1000
    fps = 1.0 / (total_infer_time / total_samples)

    print(f"\n🏆 Test set results ({DATASET } - {SUPERVISION_MODE .upper ()} DS) 🏆")
    print(f"🔹 Dice     : {np .mean (test_res ['dice']):.4f}")
    print(f"🔹 IoU      : {np .mean (test_res ['iou']):.4f}")
    print(f"🔹 ACC      : {np .mean (test_res ['acc']):.4f}")
    print(f"🔹 Precision: {np .mean (test_res ['pre']):.4f}")
    print(f"🔹 Recall   : {np .mean (test_res ['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ Inference speed evaluation ⚡")
    print(f"⏱️Average time:{avg_time_per_image :.2f} ms / image")
    print(f"🚀 F P S    : {fps :.2f} frames / second")
    print("=" * 50)
    return test_res


if __name__ == "__main__":
    main()
