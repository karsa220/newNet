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
# 1. 基础模块定义 (含修好的 LeakyRCAB 与 FAM)
# ==========================================

class LeakyRCAB(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.LConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(in_channels // 4, 1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        feat = self.LConv(x)
        att = self.channel_att(self.gap(feat))
        return feat * att + identity


def get_gn_groups(channels):
    for g in [32, 16, 8, 4, 2]:
        if channels % g == 0:
            return g
    return 1


class FeatureAlignmentModule(nn.Module):
    """自适应特征对齐模块 (FAM)."""
    def __init__(self, channels):
        super().__init__()
        self.align_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_gn_groups(channels), channels),
            nn.LeakyReLU(0.01, inplace=True)
        )

    def forward(self, low_res, high_res):
        # 将低分辨率特征上采样到与高分辨率特征一致的大小
        up_low = F.interpolate(low_res, size=high_res.shape[2:], mode='bilinear', align_corners=False)
        # 拼接并在通道层面做动态特征对齐
        fused = torch.cat([up_low, high_res], dim=1)
        out = self.align_conv(fused)
        return out


class PixelWiseCrossScaleAttention(nn.Module):
    """基于像素级的空间-尺度注意力"""
    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid_channels = max(1, (channels * num_scales) // reduction)

        self.spatial_mlp = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels * num_scales, kernel_size=1, bias=False)
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        concat_f = torch.cat(features, dim=1)
        attn_weights = self.spatial_mlp(concat_f)
        attn_weights = attn_weights.view(B, self.num_scales, C, H, W)
        attn_weights = F.softmax(attn_weights, dim=1)
        stacked_f = torch.stack(features, dim=1)
        fused_out = (stacked_f * attn_weights).sum(dim=1)
        return fused_out


class ChannelAdapter(nn.Module):
    """通道维度对齐投影层"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        return x


def seed_everything(seed=42):
    # 1. 必须在所有 CUDA 初始化之前设置这个环境变量！
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 2. 强制 CuDNN 使用确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 3. 强制 PyTorch 算子级别使用确定性算法 (关键，解决 interpolate 反向传播的随机性)
    torch.use_deterministic_algorithms(True, warn_only=True)
# ==========================================
# 2. CLADS_Net 核心网络定义 (引入 FAM 架构)
# ==========================================

class CLADS_Net(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        self.ch0, self.ch2, self.ch3, self.ch4, self.ch_b = 64, 256, 512, 1024, 1024

        # 局部通道注意力
        self.rcab2 = LeakyRCAB(self.ch2)
        self.rcab3 = LeakyRCAB(self.ch3)
        self.rcab4 = LeakyRCAB(self.ch4)

        # 统一投影通道
        self.linear0 = ChannelAdapter(self.ch0, embed_dim)
        self.linear2 = ChannelAdapter(self.ch2, embed_dim)
        self.linear3 = ChannelAdapter(self.ch3, embed_dim)
        self.linear4 = ChannelAdapter(self.ch4, embed_dim)
        self.linear_b = ChannelAdapter(self.ch_b, embed_dim)

        # ------------------ FAM 实例化 ------------------
        # 分别为深层特征初始化 4 个对齐模块，将它们动态对齐到 embed_dim
        self.fam2 = FeatureAlignmentModule(embed_dim)
        self.fam3 = FeatureAlignmentModule(embed_dim)
        self.fam4 = FeatureAlignmentModule(embed_dim)
        self.famb = FeatureAlignmentModule(embed_dim)
        # -----------------------------------------------

        # 核心跨尺度融合
        self.csma_fuse = PixelWiseCrossScaleAttention(channels=embed_dim, num_scales=5, reduction=4)

        # 预测与深度监督
        self.final_conv = nn.Conv2d(embed_dim, 1, 1)
        self.aux0 = nn.Conv2d(embed_dim, 1, 1)
        self.aux2 = nn.Conv2d(embed_dim, 1, 1)
        self.aux3 = nn.Conv2d(embed_dim, 1, 1)
        self.aux4 = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, x):
        # Backbone 特征提取
        e0 = self.encoder.relu0(self.encoder.norm0(self.encoder.conv0(x))) # (B, 64, 128, 128)
        e2 = self.encoder.denseblock1(self.encoder.pool0(e0))             # (B, 256, 64, 64)
        e3 = self.encoder.denseblock2(self.encoder.transition1(e2))       # (B, 512, 32, 32)
        e4 = self.encoder.denseblock3(self.encoder.transition2(e3))       # (B, 1024, 16, 16)
        b = self.encoder.norm5(self.encoder.denseblock4(self.encoder.transition3(e4))) # (B, 1024, 8, 8)

        # 局部 RCAB 增强
        s2 = self.rcab2(e2)
        s3 = self.rcab3(e3)
        s4 = self.rcab4(e4)

        # 1. 浅层基础通道变换
        l0 = self.linear0(e0) # 作为高分辨率锚点 (B, 256, 128, 128)

        # 2. ------------------ 使用 FAM 进行特征对齐 ------------------
        # 不再使用简单的 F.interpolate，而是让 FAM 根据浅层高精特征自主对齐边缘
        l2 = self.fam2(self.linear2(s2), l0)
        l3 = self.fam3(self.linear3(s3), l0)
        l4 = self.fam4(self.linear4(s4), l0)
        lb = self.famb(self.linear_b(b), l0)
        # ------------------------------------------------------------

        # 将动态特征对齐后的 5 个尺度特征喂给注意力融合模块
        fused = self.csma_fuse(l0, l2, l3, l4, lb)

        out = torch.sigmoid(
            F.interpolate(self.final_conv(fused), size=x.shape[2:], mode='bilinear', align_corners=False))

        if self.training:
            out0 = torch.sigmoid(F.interpolate(self.aux0(l0), size=x.shape[2:], mode='bilinear', align_corners=False))
            out1 = torch.sigmoid(F.interpolate(self.aux2(l2), size=x.shape[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux3(lb), size=x.shape[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux4(l4), size=x.shape[2:], mode='bilinear', align_corners=False))
            return out, out0, out1, out2, out3
        return out


# ==========================================
# 3. 损失函数与评估指标
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
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth)
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
    dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    acc = (tp + tn + smooth) / (tp + fp + fn + tn + smooth)
    pre = (tp + smooth) / (tp + fp + smooth)
    rec = (tp + smooth) / (tp + fn + smooth)
    return {"dice": dice.tolist(), "iou": iou.tolist(), "acc": acc.tolist(), "pre": pre.tolist(), "rec": rec.tolist()}


# ==========================================
# 4. 数据加载与主流程
# ==========================================

class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, is_train=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256))

        if self.is_train:
            # 1. 随机水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 2. 随机旋转
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # 3. 增加一点尺度缩放和平移 (超声图像中病灶大小差异很大)
            if random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                translate = [int(random.uniform(-10, 10)), int(random.uniform(-10, 10))]
                image = TF.affine(image, angle=0, translate=translate, scale=scale, shear=0)
                mask = TF.affine(mask, angle=0, translate=translate, scale=scale, shear=0)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)


        return image, mask


def get_dataset_paths(data_dir):
    image_paths, mask_paths = [], []
    categories = ['benign', 'malignant']
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.exists(cat_dir): continue

        # ⚠️ 【关键修复 1】：加上 sorted() 保证初始顺序永远固定！
        for file_name in sorted(os.listdir(cat_dir)):
            if file_name.endswith('.png') and not file_name.endswith('_mask.png'):
                img_path = os.path.join(cat_dir, file_name)
                mask_path = os.path.join(cat_dir, file_name.replace('.png', '_mask.png'))
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
    return image_paths, mask_paths

# ⚠️ 新增 worker 种子初始化函数
def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
def main():
    set_seed(42)
    seed_everything(42)
    # ⚠️ 【新增】：创建一个独立的随机数生成器并绑定种子
    g = torch.Generator()
    g.manual_seed(42)
    MODE = "train"
    save_path = "best_busi.pth"
    data_dir = r"../../data/Dataset_BUSI_with_GT"  # 请根据实际修改

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 📐 Architecture: CLADS-net + FAM")

    all_imgs, all_masks = get_dataset_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0: return

    combined = list(zip(all_imgs, all_masks))

    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    # 实例化搭载了 FAM 的新网络
    model = CLADS_Net(embed_dim=256).to(device)

    if MODE == "train":
        # ⚠️ 【修改】：传入 generator 参数，彻底锁死 Shuffle 的顺序
        train_loader = DataLoader(
            BUSIDataset(all_imgs[:train_size], all_masks[:train_size], True),
            batch_size=8,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,  # <--- ⚠️ 绑定 worker_init_fn
            num_workers=0
        )

        # 验证集和测试集不需要 shuffle，默认即可
        val_loader = DataLoader(
            BUSIDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size], False),
            batch_size=8
        )

        criterion = HybridLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        num_epochs = 35
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_dice = 0.0
        for epoch in range(num_epochs):
            model.train()
            train_loss_epoch = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for images, masks in pbar:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                out, out0, out1, out2, out3 = model(images)
                loss_main = criterion(out, masks)
                loss_aux = (
                        0.4 * criterion(out0, masks) +
                        0.4 * criterion(out1, masks) +
                        0.4 * criterion(out2, masks) +
                        0.4 * criterion(out3, masks)
                )

                loss = loss_main + loss_aux

                loss.backward()
                optimizer.step()
                train_loss_epoch += loss.item()
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

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
                f"📉 Epoch {epoch + 1} | Loss: {train_loss_epoch / len(train_loader):.4f} | Val Dice: {avg_val_dice:.4f}")

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), save_path)
                print(f"🌟 Saved Best Model: {best_val_dice:.4f}")

    print("\n" + "=" * 50)
    print("🚀 开始在完全未见的【测试集 (Test Set)】上评估最终性能...")

    test_loader = DataLoader(BUSIDataset(all_imgs[train_size + val_size:], all_masks[train_size + val_size:], False),
                             batch_size=8, shuffle=False)

    if not os.path.exists(save_path):
        print(f"❌ 找不到权重文件: {save_path}！请先运行 train 模式训练。")
        return None

    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    print("🔥 正在进行 GPU 预热 (Warm-up)...")
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

            total_infer_time += (end_time - start_time)
            total_samples += images.size(0)

            metrics = calculate_metrics(outputs, masks)
            for k in test_res.keys():
                test_res[k].extend(metrics[k])

    avg_time_per_image = (total_infer_time / total_samples) * 1000
    fps = 1.0 / (total_infer_time / total_samples)
    print("\n🏆 CLADS_Net + FAM 最终测试集成绩 🏆")
    print(f"🔹 Dice     : {np.mean(test_res['dice']):.4f}")
    print(f"🔹 IoU      : {np.mean(test_res['iou']):.4f}")
    print(f"🔹 ACC      : {np.mean(test_res['acc']):.4f}")
    print(f"🔹 Precision: {np.mean(test_res['pre']):.4f}")
    print(f"🔹 Recall   : {np.mean(test_res['rec']):.4f}")
    print("-" * 50)
    print(f"⚡ 推理速度评估 (Device: {device}) ⚡")
    print(f"⏱️ 平均耗时 : {avg_time_per_image:.2f} ms / image")
    print(f"🚀 F P S    : {fps:.2f} frames / second")
    print("=" * 50)
    return test_res


if __name__ == "__main__":
    main()