from seed.set_seed import set_seed
import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Import MISSFormer and wrap it with Sigmoid
# ==========================================
from networks.MISSFormer import MISSFormer


class MISSFormerWrapper(nn.Module):
    def __init__(self, n_classes=1):
        super(MISSFormerWrapper, self).__init__()
        self.model = MISSFormer(num_classes=n_classes)

    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits)

        # ==========================================
        # 2. Modified binary loss and index calculation
        # ==========================================


class AuthorStyleLoss(nn.Module):
    def __init__(self):
        super(AuthorStyleLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        loss_bce = self.bce(pred, target)
        smooth = 1e-5
        intersect = torch.sum(pred * target)

        # 🚀 Must be corrected to the standard binary Dice denominator to prevent the gradient from disappearing due to squares
        dice = (2 * intersect + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
        loss_dice = 1 - dice

        return 0.4 * loss_bce + 0.6 * loss_dice


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
    # 3. Dataset loading
    # ==========================================


class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = (
            image_paths,
            mask_paths,
            is_train,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.is_train:
            # 1. Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                # 2. Random rotation
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # 3. Add a little scaling and translation (lesion sizes vary greatly in ultrasound images)
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


def get_udiat_paths(data_dir):
    img_dir, gt_dir = os.path.join(data_dir, "original"), os.path.join(data_dir, "GT")
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"❌ Path error: original or GT folder not found in{data_dir }")
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

    print(f"✅ Found successfully{len (img_p )}For UDIAT images and masks.")
    return img_p, mask_p

    # ==========================================
    # 4. Main console
    # ==========================================


def main():
    set_seed(42)
    # 🌟🌟🌟 Control Panel 🌟🌟🌟
    MODE = "train"  # First set it to train and run once
    data_dir = r"../../data/UDIAT_Dataset_B"

    save_path = "best_UDIAT.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device } | ⚙️ Mode: {MODE }")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    if not all_imgs:
        return

    combined = list(zip(all_imgs, all_masks))

    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total = len(all_imgs)
    t_size, v_size = int(0.8 * total), int(0.1 * total)

    model = MISSFormerWrapper(n_classes=1).to(device)

    if MODE == "train":

        train_loader = DataLoader(
            UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True),
            batch_size=8,
            shuffle=True,
        )
        val_loader = DataLoader(
            UDIATDataset(
                all_imgs[t_size : t_size + v_size],
                all_masks[t_size : t_size + v_size],
                False,
            ),
            batch_size=8,
            shuffle=False,
        )

        criterion = AuthorStyleLoss()

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 10
        no_improve_count = 0
        history_loss, history_val_dice = [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(
                train_loader, desc=f"Epoch [{epoch +1 }/{num_epochs }]", leave=False
            )

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()

                # 🚀 Gradient clipping to prevent Transformer from gradient explosion at 1e-3 learning rate
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss_epoch += loss.item()
                pbar.set_postfix(
                    {
                        "Loss": f"{loss .item ():.4f}",
                        "LR": f"{optimizer .param_groups [0 ]['lr']:.6f}",
                    }
                )

            avg_train_loss = train_loss_epoch / len(train_loader)

            # Validation assessment
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = model(images.to(device))
                    val_dices.extend(calculate_metrics(out, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch +1 } | Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f} | LR: {optimizer .param_groups [0 ]['lr']:.6f}"
            )

            scheduler.step()

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(
                    f"🌟 [New Best Saved] Val Dice: {best_val_dice :.4f} -> Saved to {save_path }"
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stop trigger: continuous{patience }It is verified that Dice has not improved after 1 epoch and the training is stopped."
                    )
                    break

            torch.cuda.empty_cache()
            gc.collect()

            # ==========================================
            # 5. Test mode (Test)
            # ==========================================
    if MODE in ["train", "test"]:
        test_loader = DataLoader(
            UDIATDataset(
                all_imgs[t_size + v_size :], all_masks[t_size + v_size :], False
            ),
            batch_size=1,
            shuffle=False,
        )
        print("\n" + "=" * 50)
        print("🚀 Start evaluating MISSFormer on a completely unseen [test set]...")

        if not os.path.exists(save_path):
            print(f"❌ Weight not found{save_path }! Please run train first.")
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
            print("🔥GPU Warm-up...")
            with torch.no_grad():
                for _ in range(5):
                    model(torch.randn(1, 3, 224, 224).to(device))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_infer_time, total_samples = 0.0, 0

            with torch.no_grad():
                for images, masks in tqdm(test_loader, desc="Testing"):
                    images, masks = images.to(device), masks.to(device)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.time()
                    outputs = model(images)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    total_infer_time += time.time() - start_time
                    total_samples += images.size(0)

                    metrics = calculate_metrics(outputs, masks)
                    for k in test_res.keys():
                        test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)

            print("\n🏆 MISSFormer (UDIAT Finetuned) test set final results 🏆")
            print(f"🔹 Dice     : {np .mean (test_res ['dice']):.4f}")
            print(f"🔹 IoU      : {np .mean (test_res ['iou']):.4f}")
            print(f"🔹 ACC      : {np .mean (test_res ['acc']):.4f}")
            print(f"🔹 Precision: {np .mean (test_res ['pre']):.4f}")
            print(f"🔹 Recall   : {np .mean (test_res ['rec']):.4f}")
            print("-" * 50)
            print(f"⏱️Average time:{avg_time_per_image :.2f} ms/img | FPS: {fps :.2f}")
            print("=" * 50)


if __name__ == "__main__":
    main()
