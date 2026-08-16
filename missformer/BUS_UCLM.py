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

# ==========================================
# 1. Import MISSFormer from Github source code
# ==========================================
# Please ensure that the current script is at the same level as the networks folder in the MISSFormer repository
from networks.MISSFormer import MISSFormer


class MISSFormerWrapper(nn.Module):
    """Wrapper: used to activate the output of MISSFormer through Sigmoid,
    To seamlessly adapt to BCELoss (HybridLoss) designed for two categories below."""

    def __init__(self, n_classes=1):
        super(MISSFormerWrapper, self).__init__()
        # Instantiate MISSFormer
        self.model = MISSFormer(num_classes=n_classes)

    def forward(self, x):
        logits = self.model(x)
        # Use sigmoid activation to fit binary probability output
        return torch.sigmoid(logits)

        # ==========================================
        # 2. Loss function and index calculation in the original author’s style
        # ==========================================


class AuthorStyleLoss(nn.Module):
    """Fully follows the Poly Loss weight in the original author trainer.py: 0.4 * CE + 0.6 * Dice
    Since this task is a binary classification, BCE is used instead of CE for equivalent conversion.
    """

    def __init__(self):
        super(AuthorStyleLoss, self).__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        # 1. 0.4 * BCE Loss (equivalent to binary CrossEntropy)
        loss_bce = self.bce(pred, target)

        # 2. 0.6 * Dice Loss (using the original author’s Dice calculation logic in utils.py: smooth=1e-5)
        smooth = 1e-5
        intersect = torch.sum(pred * target)
        y_sum = torch.sum(target * target)
        # Replace the original author's z_sum = torch.sum(pred * pred) with the standard one:
        dice = (2 * intersect + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)

        loss_dice = 1 - dice

        # Original author combination ratio
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
    # 3. Dataset and enhanced loading (size modified to 224 by the original author)
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

        # MISSFormer feature map scale is strongly bound to 224x224
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224))

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # The original work of MISSFormer uses Normalize([0.5], [0.5])
        image = TF.to_tensor(image)

        return image, TF.to_tensor(mask)


def get_dataset_paths(data_dir):
    image_paths, mask_paths = [], []
    categories = ["benign", "malign"]
    masks_dir = os.path.join(data_dir, "masks")
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir):
            continue
        for file_name in os.listdir(cat_dir):
            if file_name.endswith(".png"):
                img_path = os.path.join(cat_dir, file_name)
                mask_path = os.path.join(masks_dir, file_name)
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
    save_path = "best_bus_uclm.pth"

    data_dir = r"../../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
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
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

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
    # 3. Initialize MISSFormer
    # ==========================================
    # ==========================================
    # 3. Initialize MISSFormer
    # ==========================================
    model = MISSFormerWrapper(n_classes=1).to(device)

    criterion = AuthorStyleLoss()

    # ==========================================
    # 4. Training mode (Train)
    # ==========================================
    # ==========================================
    # 4. Training mode (Train) - extremely fast convergence strategy for 35 Epoch
    # ==========================================
    if MODE == "train":
        num_epochs = 1
        best_val_dice = 0.0
        patience = 5
        no_improve_count = 0

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        history_loss, history_val_dice, epochs_list = [], [], []

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0

            pbar_train = tqdm(
                train_loader,
                desc=f"Epoch [{epoch +1 }/{num_epochs }] Train",
                leave=False,
            )
            for i_batch, (images, masks) in enumerate(pbar_train):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()

                # Optional: Add gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss_epoch += loss.item()
                pbar_train.set_postfix(
                    {
                        "Loss": f"{loss .item ():.4f}",
                        "LR": f"{optimizer .param_groups [0 ]['lr']:.6f}",
                    }
                )

            avg_train_loss = train_loss_epoch / len(train_loader)

            # ... [The evaluation verification code remains unchanged] ...

            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    val_dices.extend(calculate_metrics(outputs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)

            print(
                f"📉 Epoch {epoch +1 } | Avg Train Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f} | Current LR: {optimizer .param_groups [0 ]['lr']:.6f}"
            )

            scheduler.step()

            # Update drawing data
            epochs_list.append(epoch + 1)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            # Save the best model
            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Model Saved] Val Dice: {best_val_dice :.4f}")
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
            # 5. Test mode (Test) / full index evaluation
            # ==========================================
    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print(
            "🚀 Start evaluating the final performance of MISSFormer on a completely unseen [Test Set]..."
        )

        if not os.path.exists(save_path):
            print(
                f"❌ Weight file not found:{save_path }! Please run train mode training first."
            )
            return None
        else:
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

            print("\n🏆 MISSFormer (BUSI dataset) final test set results 🏆")
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
