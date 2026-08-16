from seed.set_seed import set_seed
import os
import random
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. Core import: use official modules directly
# ==========================================
# Import official MK-UNet model
from mkunet_network import MK_UNet

# Import the official weighted structural loss function designed for medical segmentation
from train_polyp import structure_loss


# ==========================================
# 2. Indicator calculation
# ==========================================
def calculate_metrics(pred_probs, target, smooth=1e-5):
    # Note: pred_probs here must be the probability map (0~1) after sigmoid activation
    pred_bin = (pred_probs > 0.5).float()
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
    # 3. Dataset and enhanced loading (use your original logic)
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

        return image, mask


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
    MODE = "train"  # After training, you can change it to "test" to test only.
    save_path = "best_bus_uclm.pth"

    # Please make sure the path is correct
    data_dir = r"../../data/BUS-UCLM Breast ultrasound lesion segmentation dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device } | 🔄 Current Mode: {MODE .upper ()}")

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

    # The batch_size usually used by the official code is too small (such as 8). You can adjust it according to the video memory.
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(
        f"✅ Data loading completed | Train:{len (train_dataset )} | Val: {len (val_dataset )} | Test: {len (test_dataset )}"
    )

    # ==========================================
    # Instantiate the official model
    # ==========================================
    model = MK_UNet(num_classes=1, in_channels=3).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧩Official MK-UNet model parameters:{total_params /1e6 :.4f} M")

    if MODE == "train":
        # The default learning rate in the official code is 0.0005, using AdamW
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 5
        no_improve_count = 0

        history_loss, history_val_dice, epochs_list = [], [], []

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

                # The official model returns a list, we take the main output
                logits = outputs[0] if isinstance(outputs, list) else outputs

                # The official structure_loss contains BCEWithLogits internally, so logits are passed in directly.
                loss = structure_loss(logits, masks)
                loss.backward()

                # Gradient clipping is used in the official code to prevent explosions
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                optimizer.step()

                train_loss_epoch += loss.item()
                pbar_train.set_postfix({"Loss": f"{loss .item ():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # --- Validation stage ---
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    logits = outputs[0] if isinstance(outputs, list) else outputs

                    # Before calculating Dice, logits need to be manually converted to probabilities (Sigmoid)
                    probs = torch.sigmoid(logits)
                    val_dices.extend(calculate_metrics(probs, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            print(
                f"📉 Epoch {epoch +1 } | Avg Train Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f} | LR: {scheduler .get_last_lr ()[0 ]:.6f}"
            )

            epochs_list.append(epoch + 1)
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(
                    f"🌟 [New Best Official MK-UNet Saved] Val Dice: {best_val_dice :.4f}"
                )
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(
                        f"⏹️ Early stop trigger: continuous{patience }It is verified that Dice has not improved after 1 epoch and the training is stopped."
                    )
                    break

    if MODE in ["train", "test"]:
        print("\n" + "=" * 50)
        print(
            "🚀 Start evaluating Official MK-UNet on a completely unseen [Test Set]..."
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

            # GPU warm-up for accurate speed measurement
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
                    logits = outputs[0] if isinstance(outputs, list) else outputs
                    probs = torch.sigmoid(logits)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    end_time = time.time()

                    total_infer_time += end_time - start_time
                    total_samples += images.size(0)

                    metrics = calculate_metrics(probs, masks)
                    for k in test_res.keys():
                        test_res[k].extend(metrics[k])

            avg_time_per_image = (total_infer_time / total_samples) * 1000
            fps = 1.0 / (total_infer_time / total_samples)

            print("\n🏆 Official MK-UNet (BUSI dataset) final test set results 🏆")
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
