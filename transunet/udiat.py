import os
import time
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from main import calculate_metrics
from main import TransUNetWrapper

# ==========================================
# 🌟 新增: 导入 TransUNet 官方网络结构
# 确保你已经运行了 pip install ml_collections
# 并且把官方项目中的 `networks` 文件夹放到了同级目录！
# ==========================================
try:
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
except ImportError:
    raise ImportError(
        "❌ 找不到 TransUNet 代码！请先 git clone https://github.com/Beckschen/TransUNet，然后将其中的 networks 文件夹与本脚本放在同一目录下，并安装 ml_collections。")


# ==========================================
# 0. 辅助函数 (假设你已有 HybridLoss, calculate_metrics, set_seed)
# ==========================================
# from your_utils import HybridLoss, calculate_metrics, set_seed
# 注意：请确保这些函数在你的环境里存在，这里省略具体实现以免冗长

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# ==========================================
# 6. UDIAT 数据集加载 (保持你的代码不变)
# ==========================================
class UDIATDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths, self.mask_paths, self.is_train = image_paths, mask_paths, is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

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
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask


def get_udiat_paths(data_dir):
    img_dir, gt_dir = os.path.join(data_dir, 'original'), os.path.join(data_dir, 'GT')
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        print(f"❌ 路径错误: 找不到 original 或 GT 文件夹于 {data_dir}")
        return [], []

    img_p, mask_p = [], []
    for f in sorted(os.listdir(img_dir)):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path, mask_path = os.path.join(img_dir, f), os.path.join(gt_dir, f)
            if os.path.exists(mask_path):
                img_p.append(img_path)
                mask_p.append(mask_path)
            else:
                alt_f = f.replace('.PNG', '.png') if '.PNG' in f else f.replace('.png', '.PNG')
                alt_mask_path = os.path.join(gt_dir, alt_f)
                if os.path.exists(alt_mask_path):
                    img_p.append(img_path)
                    mask_p.append(alt_mask_path)

    print(f"✅ 成功找到 {len(img_p)} 对 UDIAT 图像和掩码。")
    return img_p, mask_p


# ==========================================
# 7. 主控台
# ==========================================
def main():
    set_seed(42)
    # 🌟🌟🌟 控制面板 🌟🌟🌟
    MODE = "train"  # "train" 训练(微调) | "test" 直接评估
    data_dir = r"../../data/UDIAT_Dataset_B"  # 你的 UDIAT 数据集根目录
    save_path = "best_UDIAT.pth"  # 为了与之前的SegFormer区分，改了名字

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device} | ⚙️ Mode: {MODE}")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    if not all_imgs: return

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    total = len(all_imgs)
    t_size, v_size = int(0.8 * total), int(0.1 * total)

    # 🚀 初始化 TransUNet
    model = TransUNetWrapper(img_size=256, num_classes=1).to(device)

    if MODE == "train":
        # ⚠️ 注意: TransUNet 比较大，如果你的显存不够，请把 batch_size=8 改成 4 甚至 2
        train_loader = DataLoader(UDIATDataset(all_imgs[:t_size], all_masks[:t_size], True), batch_size=8, shuffle=True)
        val_loader = DataLoader(
            UDIATDataset(all_imgs[t_size:t_size + v_size], all_masks[t_size:t_size + v_size], False), batch_size=8,
            shuffle=False)

        # 请确保你定义了 HybridLoss
        from main import HybridLoss
        criterion = HybridLoss()

        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        num_epochs = 200
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

        best_val_dice = 0.0
        patience = 5
        no_improve_count = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

            avg_train_loss = train_loss_epoch / len(train_loader)

            # 验证评估
            model.eval()
            val_dices = []
            with torch.no_grad():
                for images, masks in val_loader:
                    out = model(images.to(device))
                    if isinstance(out, tuple): out = out[0]
                    # 请确保你的 calculate_metrics 函数中做了 sigmoid 操作，因为模型输出的是 Logits
                    val_dices.extend(calculate_metrics(out, masks.to(device))["dice"])

            avg_val_dice = np.mean(val_dices)
            scheduler.step()

            print(
                f"📉 Epoch {epoch + 1} | Loss: {avg_train_loss:.4f} | Val Dice: {avg_val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 [New Best Saved] Val Dice: {best_val_dice:.4f} -> Saved to {save_path}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"⏹️ 早停触发: 连续 {patience} 个 epoch 验证 Dice 无提升，停止训练。")
                    break

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的测试集上评估...")
    test_loader = DataLoader(UDIATDataset(all_imgs[t_size + v_size:], all_masks[t_size + v_size:], False),
                             batch_size=1, shuffle=False)

    if not os.path.exists(save_path):
        print(f"❌ 找不到权重文件 {save_path}！请先运行 train 模式。")
        return

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    print("🔥 GPU 预热 (Warm-up)...")
    with torch.no_grad():
        for _ in range(5): model(torch.randn(1, 3, 256, 256).to(device))
    if torch.cuda.is_available(): torch.cuda.synchronize()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    total_time, total_samples = 0.0, 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            if torch.cuda.is_available(): torch.cuda.synchronize()

            total_time += (time.time() - t0)
            total_samples += images.size(0)

            if isinstance(outputs, tuple): outputs = outputs[0]

            m = calculate_metrics(outputs, masks)
            for k in test_res: test_res[k].extend(m[k])

    print("\n🏆 TransUNet (UDIAT Finetuned) 测试集最终成绩 🏆")
    print(f"🔹 Dice : {np.mean(test_res['dice']):.4f} | IoU: {np.mean(test_res['iou']):.4f}")
    print(
        f"🔹 ACC  : {np.mean(test_res['acc']):.4f}  | Pre: {np.mean(test_res['pre']):.4f} | Rec: {np.mean(test_res['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ 推理速度评估 (Device: {device}) ⚡")
    print(
        f"⏱️ 平均耗时: {(total_time / total_samples) * 1000:.2f} ms/img | FPS: {1.0 / (total_time / total_samples):.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()