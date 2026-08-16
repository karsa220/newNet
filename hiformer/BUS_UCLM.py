from seed.set_seed import set_seed
import os
import random
import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


from models.HiFormer import HiFormer
from configs.HiFormer_configs import (
    get_hiformer_b_configs,
    get_hiformer_s_configs,
    get_hiformer_l_configs,
)


# ==========================================
# 1. Evaluation indicators and loss functions
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_dice=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
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
    # 2. Datasets and loaders
    # ==========================================


class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        # [Repair] The is_train on the right side of the equal sign is added here.
        self.image_paths, self.mask_paths, self.is_train = (
            image_paths,
            mask_paths,
            is_train,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = TF.resize(Image.open(self.image_paths[idx]).convert("RGB"), (224, 224))
        mask = TF.resize(Image.open(self.mask_paths[idx]).convert("L"), (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            angle = random.uniform(-15, 15)
            img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)

        return TF.to_tensor(img), TF.to_tensor(mask)


def get_dataset_paths(data_dir):
    img_p, mask_p = [], []
    masks_dir = os.path.join(data_dir, "masks")
    for cat in ["benign", "malign"]:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            continue
        for f in os.listdir(cat_dir):
            if f.endswith(".png"):
                m_path = os.path.join(masks_dir, f)
                if os.path.exists(m_path):
                    img_p.append(os.path.join(cat_dir, f))
                    mask_p.append(m_path)
    return img_p, mask_p

    # ==========================================
    # 3. Main console
    # ==========================================


def main():
    set_seed(42)
    # 🌟🌟🌟 Control Panel 🌟🌟🌟
    MODE = "train"

    data_dir = r"../../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
    save_path = "best_bus_uclm.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device } | ⚙️ Mode: {MODE }")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    if len(all_imgs) == 0:
        print("❌ No data found! Please check the path.")
        return

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total_size = len(all_imgs)
    t_size, v_size = int(0.8 * total_size), int(0.1 * total_size)

    config = (
        get_hiformer_b_configs()
    )  # Get version B's configuration (you can also change to _s_configs or _l_configs)

    # Set n_classes=1 because the mask of the BUSI dataset is a single-channel target
    model = HiFormer(config=config, img_size=224, in_chans=3, n_classes=1).to(device)

    # ==========================
    # Mode A: Training mode
    # ==========================
    if MODE == "train":
        train_loader = DataLoader(
            BUSIDataset(all_imgs[:t_size], all_masks[:t_size], True),
            batch_size=8,
            shuffle=True,
        )
        val_loader = DataLoader(
            BUSIDataset(
                all_imgs[t_size : t_size + v_size],
                all_masks[t_size : t_size + v_size],
                False,
            ),
            batch_size=8,
            shuffle=False,
        )

        criterion = HybridLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 5
        no_improve_count = 0
        history_train_loss = []
        history_val_dice = []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar_train = tqdm(
                train_loader, desc=f"Epoch [{epoch +1 }/{num_epochs }]", leave=False
            )

            for images, masks in pbar_train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                # [Modification] Obtain the official model output and activate it through Sigmoid to adapt to BCELoss
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar_train.set_postfix({"Loss": f"{loss .item ():.4f}"})

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    outputs = model(images.to(device))
                    outputs = torch.sigmoid(
                        outputs
                    )  # [Modification] Sigmoid is also required for verification
                    metrics = calculate_metrics(outputs, masks.to(device))
                    val_dices.extend(metrics["dice"])

            avg_train_loss = train_loss_epoch / len(train_loader)
            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            history_train_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch +1 } | Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f}"
            )

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best HiFormer Saved] Val Dice: {best_val_dice :.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stop trigger: continuous{patience }It is verified that Dice has not improved after 1 epoch and the training is stopped."
                    )
                    break

    print("\n" + "=" * 50)
    print("🚀 Start evaluating the final performance of HiFormer on the test set...")

    test_loader = DataLoader(
        BUSIDataset(all_imgs[t_size + v_size :], all_masks[t_size + v_size :], False),
        batch_size=10,
        shuffle=False,
    )

    if not os.path.exists(save_path):
        print(
            f"❌ Weight file not found:{save_path }! Please run train mode training first."
        )
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}

    print("🔥 GPU warm-up (Warm-up) in progress...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
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

            outputs = model(images)
            outputs = torch.sigmoid(
                outputs
            )  # [Modification] Sigmoid is also required to extract the probability value during testing.

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
    print("\n🏆 HiFormer final test set results 🏆")
    print(f"🔹 Dice (F1): {np .mean (test_res ['dice']):.4f}")
    print(f"🔹 IoU      : {np .mean (test_res ['iou']):.4f}")
    print(f"🔹 ACC      : {np .mean (test_res ['acc']):.4f}")
    print(f"🔹 Precision: {np .mean (test_res ['pre']):.4f}")
    print(f"🔹 Recall   : {np .mean (test_res ['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ Inference speed evaluation (Device:{device }) ⚡")
    print(f"⏱️Average time:{avg_time_per_image :.2f} ms / image")
    print(f"🚀 F P S    : {fps :.2f} frames / second")
    print("=" * 50)
    return test_res


if __name__ == "__main__":
    main()
