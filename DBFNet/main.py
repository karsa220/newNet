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
# 1. DBF-Net (Xu et al., IEEE ICIP 2025)
#    Official code: https://github.com/apple1986/DBF-Net
#    Dual-branch body/boundary decoder with learnable fusion weights.
# ==========================================
class double_conv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch, dilation=1, re_mi=False):
        super(double_conv, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, re_mi=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, dilation=dilation, re_mi=re_mi),
        )

    def forward(self, x):
        return self.mpconv(x)


class Connection(nn.Module):
    def __init__(self):
        super(Connection, self).__init__()

    def forward(self, x1, body):
        diffY = x1.size()[2] - body.size()[2]
        diffX = x1.size()[3] - body.size()[3]
        body = F.pad(
            body,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        return torch.cat([body, x1], dim=1)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class Body_Boundary(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Body_Boundary, self).__init__()
        self.connect = Connection()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.body = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.boundary = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, encoder_x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.connect(encoder_x, x)
        x = self.conv1(x)
        body = self.body(x)
        boundary = self.boundary(x)
        return body, boundary


class Mynet(nn.Module):
    """Verbatim DBF-Net backbone. Returns 5 full-resolution heads."""

    def __init__(self, classes, re_mi=False):
        super(Mynet, self).__init__()
        self.inc = inconv(3, 32, dilation=1)
        self.down1 = down(32, 64, dilation=2, re_mi=re_mi)
        self.down2 = down(64, 128, dilation=3, re_mi=re_mi)
        self.down3 = down(128, 256, dilation=5, re_mi=re_mi)
        self.down4 = down(256, 256, dilation=7, re_mi=re_mi)

        self.BB1 = Body_Boundary(512, 128)
        self.BB2 = Body_Boundary(256, 64)
        self.BB3 = Body_Boundary(128, 32)
        self.BB4 = Body_Boundary(64, 32)
        self.outc1_boundary = outconv(128, classes)
        self.outc1_body = outconv(128, classes)
        self.outc2_boundary = outconv(64, classes)
        self.outc2_body = outconv(64, classes)
        self.outc3_boundary = outconv(32, classes)
        self.outc3_body = outconv(32, classes)
        self.outc4_boundary = outconv(32, classes)
        self.outc4_body = outconv(32, classes)
        self.outc4_main = outconv(32, classes)

        self.w1 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)

        self.w11 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w22 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w11.data.fill_(0.5)
        self.w22.data.fill_(0.5)

        self.w111 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w222 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w111.data.fill_(0.5)
        self.w222.data.fill_(0.5)

        self.w1111 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2222 = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1111.data.fill_(0.5)
        self.w2222.data.fill_(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        body1, boundary1 = self.BB1(x5, x4)
        body2, boundary2 = self.BB2(self.w1 * body1 + self.w2 * boundary1, x3)
        body3, boundary3 = self.BB3(self.w11 * body2 + self.w22 * boundary2, x2)
        body4, boundary4 = self.BB4(self.w111 * body3 + self.w222 * boundary3, x1)

        body3 = F.interpolate(body3, scale_factor=2, mode="bilinear", align_corners=True)
        boundary3 = F.interpolate(
            boundary3, scale_factor=2, mode="bilinear", align_corners=True
        )
        body3 = self.outc3_body(body3)
        boundary3 = self.outc3_boundary(boundary3)

        x = self.outc4_main(self.w1111 * body4 + self.w2222 * boundary4)
        body4 = self.outc4_body(body4)
        boundary4 = self.outc4_boundary(boundary4)
        return [body4, boundary4, x], [body3, boundary3]


class DBFNet(nn.Module):
    """Adapter for the common evaluation protocol.

    forward() returns the fused main-head probability map (sigmoid applied),
    so the shared loss/metrics code can be reused unchanged. The four
    auxiliary body/boundary heads are exposed through `aux_loss()`, which
    reproduces the official multi-head supervision.
    """

    def __init__(self, in_channels=3, num_classes=1):
        super(DBFNet, self).__init__()
        self.net = Mynet(classes=num_classes)
        self._aux = []

    def forward(self, x):
        heads, aux = self.net(x)
        body4, boundary4, main = heads
        self._aux = [body4, boundary4, aux[0], aux[1]]
        return torch.sigmoid(main)

    def aux_loss(self, criterion, target):
        if not self._aux:
            return 0.0
        losses = [criterion(torch.sigmoid(h), target) for h in self._aux]
        return sum(losses) / len(losses)

# ==========================================
# 2. Hybrid loss function and indicator calculation
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
    # 3. Model and loss function initialization (DBF-Net, ICIP 2025)
    # ==========================================
    model = DBFNet(in_channels=3, num_classes=1).to(device)
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
                loss = criterion(outputs, masks) + 0.5 * model.aux_loss(criterion, masks)
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
            "🚀 Start evaluating the final performance of Baseline (DBF-Net) on a completely unseen [Test Set]..."
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

            print("\n🏆 DBF-Net (BUSI dataset) final test set results 🏆")
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
