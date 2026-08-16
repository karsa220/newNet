from seed.set_seed import set_seed
import os
import random
import time
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

# ==========================================
# Innovation: Dynamic deep supervision weight calculation with data set size awareness
# ==========================================
import math


def calculate_ds_weight(dataset_size, N_min=130.0, N_max=624.0, W_min=0.15, W_max=0.30):
    """Use a generalized logistic function to calculate deep supervision weights based on the dataset size.
    Advantages:
    1. Boundedness: No matter how big or small the dataset_size is, the weight is always strictly limited to (W_min, W_max).
    2. Smoothness: smooth transition to avoid sudden changes in linear functions at the boundary.
    """
    # 1. Calculate the inflection point mu (i.e. the midpoint of N_min and N_max)
    mu = (N_min + N_max) / 2.0

    # 2. Calculate growth rate gamma
    # We hope that at N_max, the function value reaches 95% of the transition completion of the upper bound
    # Mathematical derivation: 1 / (1 + exp(-gamma * (N_max - mu))) = 0.95 => gamma = ln(19) / (N_max - mu)
    gamma = math.log(19) / (N_max - mu)

    # 3. Calculate logistic mapping weight
    weight = W_min + (W_max - W_min) / (1.0 + math.exp(-gamma * (dataset_size - mu)))

    return weight

    # Dual-Pooling Residual Channel Attention Block


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

        # Optimization 2: Introducing double pooling (Average + Max)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        mid = max(in_channels // 4, 1)

        # Channel Attention MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        feat = self.LConv(x)
        avg_att = self.mlp(self.gap(feat))
        max_att = self.mlp(self.gmp(feat))
        att = self.sigmoid(avg_att + max_att)
        out = feat * att + identity
        return out


# acssr: Adaptive Cross-Scale Soft Routing
class AdaptiveCrossScaleSoftRouting(nn.Module):
    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid_channels = max(1, (channels * num_scales) // reduction)

        self.router = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels * num_scales, kernel_size=1, bias=False),
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        concat_f = torch.cat(features, dim=1)
        attn_weights = self.router(concat_f)
        attn_weights = attn_weights.view(B, self.num_scales, C, H, W)
        attn_weights = F.softmax(attn_weights, dim=1)
        stacked_f = torch.stack(features, dim=1)
        fused_out = (stacked_f * attn_weights).sum(dim=1)
        return fused_out


# Projection layer responsible for unifying channel dimensions
class ChannelAdapter(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


# Dual-Pooling Residual Adaptive Soft-Routing Network
class DPRASR_Net(nn.Module):
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

        self.acssr = AdaptiveCrossScaleSoftRouting(
            channels=embed_dim, num_scales=5, reduction=4
        )

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

        s1 = self.rcab1(e1)
        s2 = self.rcab2(e2)
        s3 = self.rcab3(e3)

        target_size = e0.shape[2:]

        l0 = self.linear0(e0)
        l1 = F.interpolate(
            self.linear1(s1), target_size, mode="bilinear", align_corners=False
        )
        l2 = F.interpolate(
            self.linear2(s2), target_size, mode="bilinear", align_corners=False
        )
        l3 = F.interpolate(
            self.linear3(s3), target_size, mode="bilinear", align_corners=False
        )
        lb = F.interpolate(
            self.linear_b(b), target_size, mode="bilinear", align_corners=False
        )

        fused = self.acssr(l0, l1, l2, l3, lb)

        out = torch.sigmoid(
            F.interpolate(
                self.final_conv(fused),
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
        )

        if self.training:
            out0 = torch.sigmoid(
                F.interpolate(
                    self.aux0(l0),
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            out1 = torch.sigmoid(
                F.interpolate(
                    self.aux1(l1),
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            out2 = torch.sigmoid(
                F.interpolate(
                    self.aux2(lb),
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            out3 = torch.sigmoid(
                F.interpolate(
                    self.aux3(l3),
                    size=x.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
            return out, out0, out1, out2, out3
        return out


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


def main():
    set_seed(42)
    MODE = "train"  # The default is train for testing
    save_path = "best_busi.pth"
    data_dir = r"../../data/Dataset_BUSI_with_GT"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device } | 📐 Architecture: DPRASR-Net")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ Dataset not found.")
        return

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # ==========================================
    # 🌟 Innovative application: dynamically calculate the current depth supervision weight
    # ==========================================
    dynamic_ds_weight = calculate_ds_weight(train_size)
    print(
        f"🎯 DSADDS: Auto-adjusted Deep Supervision Weight = {dynamic_ds_weight :.4f} (Training Size = {train_size })"
    )

    model = DPRASR_Net(embed_dim=256).to(device)

    if MODE == "train":
        train_loader = DataLoader(
            BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True),
            batch_size=8,
            shuffle=True,
        )
        val_loader = DataLoader(
            BUSIDataset(
                all_imgs[train_size : train_size + val_size],
                all_masks[train_size : train_size + val_size],
                False,
            ),
            batch_size=8,
        )

        criterion = HybridLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
            pbar = tqdm(train_loader, desc=f"Epoch {epoch +1 }/{num_epochs }")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                out, out0, out1, out2, out3 = model(images)
                loss_main = criterion(out, masks)

                loss_aux = dynamic_ds_weight * (
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

    print("\n" + "=" * 50)
    print(
        "🚀 Start evaluating the final performance on a completely unseen [Test Set]..."
    )

    test_loader = DataLoader(
        BUSIDataset(
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
    print("\n🏆 DPRASR-Net final test set results 🏆")
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
