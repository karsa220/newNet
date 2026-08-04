from seed.set_seed import set_seed
import os
import random
import time
import math
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
from DPRASRNet.rcab_dprcab_comparison import RCAB

# ==========================================
# 1. 动态深度监督权重计算 (dprcab)
# ==========================================
def calculate_ds_weight(dataset_size, N_min=130.0, N_max=624.0, W_min=0.15, W_max=0.45):
    """
    使用广义逻辑增长函数 (Logistic Function) 根据数据集大小计算深度监督权重。
    优势:
    1. 有界性: 无论 dataset_size 有多大多小，权重永远严格限制在 (W_min, W_max) 之间。
    2. 平滑性: 过渡平滑，避免线性函数在边界处的突变。
    """
    # 1. 计算拐点 mu (即 N_min 和 N_max 的中点)
    mu = (N_min + N_max) / 2.0

    # 2. 计算增长率 gamma
    # 我们希望在 N_max 时，函数值达到上界的 95% 的过渡完成度
    # 数学推导: 1 / (1 + exp(-gamma * (N_max - mu))) = 0.95  =>  gamma = ln(19) / (N_max - mu)
    gamma = math.log(19) / (N_max - mu)

    # 3. 计算 Logistic 映射权重
    weight = W_min + (W_max - W_min) / (1.0 + math.exp(-gamma * (dataset_size - mu)))

    return weight


# ==========================================
# 2. 网络核心组件定义
# ==========================================
class DPRCAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LConv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // 4, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False)
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
            nn.Conv2d(mid_channels, channels * num_scales, kernel_size=1, bias=False)
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


# ==========================================
# 3. 支持消融配置的可调网络模型
# ==========================================
class DPRASR_Net_Ablation(nn.Module):
    def __init__(self, embed_dim=256, use_dprcab=True, use_acssr=True,use_rcab=False):
        super().__init__()
        self.use_dprcab = use_dprcab
        self.use_acssr = use_acssr
        self.use_rcab = use_rcab
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = densenet.features

        self.ch0, self.ch2, self.ch3, self.ch4, self.ch_b = 64, 256, 512, 1024, 1024

        if self.use_dprcab:
            self.rcab1 = DPRCAB(self.ch2)
            self.rcab2 = DPRCAB(self.ch3)
            self.rcab3 = DPRCAB(self.ch4)
        if self.use_rcab:

            self.rcab1 = RCAB(self.ch2)
            self.rcab2 = RCAB(self.ch3)
            self.rcab3 = RCAB(self.ch4)
        self.linear0 = ChannelAdapter(self.ch0, embed_dim)
        self.linear1 = ChannelAdapter(self.ch2, embed_dim)
        self.linear2 = ChannelAdapter(self.ch3, embed_dim)
        self.linear3 = ChannelAdapter(self.ch4, embed_dim)
        self.linear_b = ChannelAdapter(self.ch_b, embed_dim)

        if self.use_acssr:
            self.acssr = AdaptiveCrossScaleSoftRouting(channels=embed_dim, num_scales=5, reduction=4)
        else:
            self.baseline_fusion = nn.Sequential(
                nn.Conv2d(embed_dim * 5, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
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

        if self.use_dprcab:
            s1 = self.rcab1(e1)
            s2 = self.rcab2(e2)
            s3 = self.rcab3(e3)
        else:
            s1, s2, s3 = e1, e2, e3

        target_size = e0.shape[2:]

        l0 = self.linear0(e0)
        l1 = F.interpolate(self.linear1(s1), target_size, mode='bilinear', align_corners=False)
        l2 = F.interpolate(self.linear2(s2), target_size, mode='bilinear', align_corners=False)
        l3 = F.interpolate(self.linear3(s3), target_size, mode='bilinear', align_corners=False)
        lb = F.interpolate(self.linear_b(b), target_size, mode='bilinear', align_corners=False)

        if self.use_acssr:
            fused = self.acssr(l0, l1, l2, l3, lb)
        else:
            fused = self.baseline_fusion(torch.cat([l0, l1, l2, l3, lb], dim=1))

        out = torch.sigmoid(
            F.interpolate(self.final_conv(fused), size=x.shape[2:], mode='bilinear', align_corners=False))

        if self.training:
            out0 = torch.sigmoid(F.interpolate(self.aux0(l0), size=x.shape[2:], mode='bilinear', align_corners=False))
            out1 = torch.sigmoid(F.interpolate(self.aux1(l1), size=x.shape[2:], mode='bilinear', align_corners=False))
            out2 = torch.sigmoid(F.interpolate(self.aux2(lb), size=x.shape[2:], mode='bilinear', align_corners=False))
            out3 = torch.sigmoid(F.interpolate(self.aux3(l3), size=x.shape[2:], mode='bilinear', align_corners=False))
            return out, out0, out1, out2, out3
        return out


# ==========================================
# 4. 损失函数与指标计算
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
# 5. UDIAT 数据集加载类
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
# 6. 单个变体的训练与评估函数
# ==========================================
def train_and_eval_variant(config, train_loader, val_loader, test_loader, device, train_size):
    name = config['name']
    use_dprcab = config['use_dprcab']
    use_acssr = config['use_acssr']
    use_rcab = config['use_rcab']

    print(f"\n" + "=" * 60)
    print(f"🚀 开始消融实验组: [{name}]")
    print(f"⚙️ 配置: DPRCAB={use_dprcab} | ACSSR={use_acssr} | RCAB={use_rcab}")
    print("=" * 60)

    ds_weight = calculate_ds_weight(train_size)

    model = DPRASR_Net_Ablation(embed_dim=256, use_dprcab=use_dprcab, use_acssr=use_acssr,use_rcab=use_rcab).to(device)
    criterion = HybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    num_epochs = 200
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)

    # 规范化保存路径文件名
    safe_name = name.replace(' ', '_').replace('&', 'and').replace('\\', '').replace('/', '_')
    save_path = f"best_udiat_ablation_{safe_name}.pth"

    best_val_dice = 0.0
    patience = 10
    no_improve_count = 0

    # ----- 训练阶段 -----
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch + 1}/{num_epochs}", leave=False)

        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()


            out, out0, out1, out2, out3 = model(images)
            loss_main = criterion(out, masks)
            loss_aux = ds_weight * (
                        criterion(out0, masks) + criterion(out1, masks) + criterion(out2, masks) + criterion(out3,
                                                                                                             masks))
            loss = loss_main + loss_aux


            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()

        model.eval()
        val_dices = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_dices.extend(calculate_metrics(outputs, masks)["dice"])

        avg_val_dice = np.mean(val_dices)
        scheduler.step()

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    # ----- 测试集评估阶段 -----
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    test_res = {"dice": [], "iou": [], "acc": [], "pre": [], "rec": []}
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            metrics = calculate_metrics(outputs, masks)
            for k in test_res.keys():
                test_res[k].extend(metrics[k])

    results = {
        "Dice": np.mean(test_res['dice']),
        "IoU": np.mean(test_res['iou']),
        "ACC": np.mean(test_res['acc']),
        "Precision": np.mean(test_res['pre']),
        "Recall": np.mean(test_res['rec']),
    }

    print(f"✅ [{name}] 测试完成! Dice: {results['Dice']:.4f} | IoU: {results['IoU']:.4f}")
    return results


# ==========================================
# 7. 主函数与 LaTeX 表格自动生成
# ==========================================
def main():
    set_seed(42)
    data_dir =r"../../data/UDIAT_Dataset_B"   # 请修改为你的 UDIAT 数据集路径

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device} | 🧪 Running Ablation Study on UDIAT Dataset")

    all_imgs, all_masks = get_udiat_paths(data_dir)
    total_size = len(all_imgs)
    if total_size == 0:
        print("❌ 未找到数据集，请核对 data_dir 路径！")
        return

    combined = list(zip(all_imgs, all_masks))
    random.shuffle(combined)
    all_imgs, all_masks = zip(*combined)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_loader = DataLoader(
        UDIATDataset(all_imgs[:train_size], all_masks[:train_size], is_train=True),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        UDIATDataset(all_imgs[train_size:train_size + val_size], all_masks[train_size:train_size + val_size],
                     is_train=False),
        batch_size=8, shuffle=False
    )
    test_loader = DataLoader(
        UDIATDataset(all_imgs[train_size + val_size:], all_masks[train_size + val_size:], is_train=False),
        batch_size=8, shuffle=False
    )

    # ==========================================
    # 采用 w/o 格式的消融配置 (符合顶级期刊习惯)
    # ==========================================
    ablation_configs = [
      {"name": r"w/o DPRCAB \& ACSSR ", "use_acssr": False, "use_dprcab": False,"use_rcab": False},

     {"name": r"w/o ACSSR ", "use_acssr": False, "use_dprcab": True,"use_rcab": False},

        {"name": r"w/o DPRCAB ",  "use_acssr": True, "use_dprcab": False,"use_rcab": False},
      {"name": r"w/o ACSSR use RCAB", "use_acssr": False, "use_dprcab": False, "use_rcab": True},
        {"name": r"w/o DPRCAB", "use_acssr": True, "use_dprcab": False,"use_rcab": False},
        {"name": r"DPRASR-Net (Ours)", "use_acssr": True, "use_dprcab": True,"use_rcab": False},  # Full Model
    ]

    all_results = {}

    for config in ablation_configs:
        res = train_and_eval_variant(config, train_loader, val_loader, test_loader, device, train_size)
        all_results[config['name']] = res

    # ==========================================
    # 生成 LaTeX 代码并写入文件
    # ==========================================
    metrics_keys = ["Dice", "IoU", "ACC", "Precision", "Recall"]
    best_vals = {m: max([res[m] for res in all_results.values()]) for m in metrics_keys}

    latex_lines = []
    latex_lines.append(
        r"\textbf{Model Configuration} & \textbf{Dice} & \textbf{IoU} & \textbf{ACC} & \textbf{Precision} & \textbf{Recall} \\")
    latex_lines.append(r"\midrule")

    for config in ablation_configs:
        name = config['name']
        res = all_results[name]

        if name == r"DPRASR-Net (Ours)":
            latex_lines.append(r"\midrule")
            row_str = [r"\textbf{DPRASR-Net (Ours)}"]
        else:
            row_str = [name]

        for m in metrics_keys:
            val = res[m]
            if abs(val - best_vals[m]) < 1e-6:
                row_str.append(rf"\textbf{{{val:.4f}}}")
            else:
                row_str.append(f"{val:.4f}")

        latex_lines.append(" & ".join(row_str) + r" \\")

    # 1. 打印到控制台
    print("\n\n" + "=" * 80)
    print("📋 复制以下 LaTeX 代码到你的论文中 (Table 4):")
    print("=" * 80 + "\n")
    for line in latex_lines:
        print(line)
    print("\n" + "=" * 80)

    # 2. 写入到文件中
    output_filename = "ablation_results.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        for line in latex_lines:
            f.write(line + "\n")

    print(f"💾 结果已成功保存至当前目录下的文件: {output_filename}")


if __name__ == "__main__":
    main()