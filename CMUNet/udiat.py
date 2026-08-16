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
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Import the replacement model as CMU-Net
# ==========================================
from src.network.CMUNet import CMUNet
from src.losses import BCEDiceLoss


# ==========================================
# 2. Indicator calculation function
# ==========================================
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
    # 3. UDIAT data set loading (original image/GT structure)
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
                # Compatible with case differences (such as .PNG and .png)
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
    MODE = "train"  # "train" training (fine-tuning) | "test" direct evaluation
    data_dir = r"../../data/UDIAT_Dataset_B"  # Your UDIAT data set root directory

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

    # 🚀 Modification 2: Initialize CMU-Net model
    print("🔥Initialize CMU-Net model...")
    model = CMUNet(img_ch=3, output_ch=1, l=7, k=7).to(device)

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

        criterion = BCEDiceLoss().to(device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=35, eta_min=1e-6
        )

        best_val_dice = 0.0
        patience = 5
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

                # 🚀 Modification 4: CMU-Net has only one final output, and Loss will automatically do Sigmoid internally
                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({"Loss": f"{loss .item ():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # Validation assessment
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    out = model(images)

                    # 🚀 Modification 5: When verifying, output the network through Sigmoid to convert to probability
                    pred = torch.sigmoid(out)
                    val_dices.extend(calculate_metrics(pred, masks)["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()
            history_loss.append(avg_train_loss)
            history_val_dice.append(avg_val_dice)

            print(
                f"📉 Epoch {epoch +1 } | Loss: {avg_train_loss :.4f} | Val Dice: {avg_val_dice :.4f} | LR: {scheduler .get_last_lr ()[0 ]:.6f}"
            )

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

                    # Static drawing of curve graph after training
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history_loss, marker="o")
        plt.title("Train Loss")
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(history_val_dice, marker="s", color="orange")
        plt.title("Val Dice")
        plt.grid()
        plt.tight_layout()
        plt.savefig("training_curve_CMUNet_UDIAT_Finetuned.png", dpi=150)
        print(
            "✅ Training is completed and the training curve has been saved as training_curve_CMUNet_UDIAT_Finetuned.png"
        )

    print("\n" + "=" * 50)
    print("🚀 Start evaluating on a completely unseen test set...")
    test_loader = DataLoader(
        UDIATDataset(all_imgs[t_size + v_size :], all_masks[t_size + v_size :], False),
        batch_size=1,
        shuffle=False,
    )

    # Testing phase: Load the latest weights saved after fine-tuning
    if not os.path.exists(save_path):
        print(f"❌ Weight file not found{save_path }! Please run train mode first.")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    print("🔥GPU Warm-up...")
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(1, 3, 256, 256).to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            total_time += time.time() - t0
            total_samples += images.size(0)

            # 🚀 Modification 6: Test inference also extracts a single output and applies a layer of Sigmoid
            pred = torch.sigmoid(outputs)
            m = calculate_metrics(pred, masks)

            for k in test_res:
                test_res[k].extend(m[k])

    print("\n🏆 CMU-Net (UDIAT Finetuned) test set final results 🏆")
    print(
        f"🔹 Dice : {np .mean (test_res ['dice']):.4f} | IoU: {np .mean (test_res ['iou']):.4f}"
    )
    print(
        f"🔹 ACC  : {np .mean (test_res ['acc']):.4f}  | Pre: {np .mean (test_res ['pre']):.4f} | Rec: {np .mean (test_res ['rec']):.4f}"
    )
    print("-" * 50)
    print(f"⚡ Inference speed evaluation (Device:{device }) ⚡")
    print(
        f"⏱️Average time:{(total_time /total_samples )*1000 :.2f} ms/img | FPS: {1.0 /(total_time /total_samples ):.2f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
