from seed.set_seed import set_seed
import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ==========================================
# 1. Ordinary U-Net model definition (new)
# ==========================================
class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        # Use sigmoid activation to adapt nn.BCELoss
        return torch.sigmoid(logits)

        # ==========================================
        # 2. Hybrid loss function and indicator calculation (retain the original version)
        # ==========================================


class HybridLoss(nn.Module):
    def __init__(self, weight_bce=0.4, weight_jaccard=0.6):
        super(HybridLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.weight_bce = weight_bce
        self.weight_jaccard = weight_jaccard

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

        jaccard_loss = 1.0 - (intersection + smooth) / (union + smooth)

        return (self.weight_bce * bce_loss) + (
            self.weight_jaccard * jaccard_loss.mean()
        )


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
    # 3. Dataset and enhanced loading (retain the original version)
    # ==========================================


class BUSIDataset(Dataset):
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

        # [Key fix] Add ImageNet Normalization
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        return image, mask


def get_dataset_paths(data_dir):
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

    # ==========================================
    # 4. Main console
    # ==========================================


def main():
    set_seed(42)
    # ==========================================
    # 1. Operation mode setting
    # ==========================================
    MODE = "train"  # Optional: "train" or "test"
    save_path = "best_busi.pth"

    data_dir = r"../../data/Dataset_BUSI_with_GT"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device } | 🔄 Current Mode: {MODE .upper ()}")

    # ==========================================
    # 2. Data set loading and partitioning
    # ==========================================
    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ No data found! Please check the path.")
        exit()

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_imgs, train_masks = all_imgs[:train_size], all_masks[:train_size]
    val_imgs, val_masks = (
        all_imgs[train_size : train_size + val_size],
        all_masks[train_size : train_size + val_size],
    )
    test_imgs, test_masks = (
        all_imgs[train_size + val_size :],
        all_masks[train_size + val_size :],
    )

    train_dataset = BUSIDataset(train_imgs, train_masks, is_train=True)
    val_dataset = BUSIDataset(val_imgs, val_masks, is_train=False)
    test_dataset = BUSIDataset(test_imgs, test_masks, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    print(
        f"✅ Data loading completed | Train:{len (train_dataset )} | Val: {len (val_dataset )} | Test: {len (test_dataset )}"
    )

    # ==========================================
    # 3. Model and loss function initialization (replaced by ordinary U-Net)
    # ==========================================
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = HybridLoss()

    # ==========================================
    # 4. Training mode (Train)
    # ==========================================
    if MODE == "train":
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        num_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 5
        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            pbar_train = tqdm(
                train_loader,
                desc=f"Epoch [{epoch +1 }/{num_epochs }] Train",
                leave=False,
            )
            for images, masks in pbar_train:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({"Loss": f"{loss .item ():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

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
                f"📉 Epoch {epoch +1 } | Avg Train Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f} | LR: {scheduler .get_last_lr ()[0 ]:.6f}"
            )

            # Save the best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Baseline Saved] Val Dice: {best_val_dice :.4f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stop trigger: continuous{patience }It is verified that Dice has not improved after 1 epoch and the training is stopped."
                    )
                    break

                    # ==========================================
                    # 5. Test mode (Test) / full index evaluation
                    # ==========================================
    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print(
            "🚀 Start evaluating the final performance of Baseline (U-Net) on a completely unseen [Test Set]..."
        )

        if not os.path.exists(save_path):
            print(
                f"❌ Weight file not found:{save_path }! Please run train mode training first."
            )
        else:
            model.load_state_dict(torch.load(save_path, map_location=device))
            model.eval()

            test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
            print("🔥 GPU warm-up (Warm-up) in progress...")
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

            print("\n🏆 UNet (BUSI dataset) final test set results 🏆")
            print(f"🔹 Dice     : {np .mean (test_res ['dice']):.4f}")
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
