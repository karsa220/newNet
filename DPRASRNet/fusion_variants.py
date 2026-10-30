"""
Controlled fusion-routing variants for the ACSSR ablation.
===========================================================

Three routing modules are strictly controlled: identical input interface
(K feature maps of shape (B, C, H, W)), identical output shape, identical
position in the network (after ChannelAdapter, before the final 1x1 conv).
Only the *routing mechanism* differs:

  V1  SKNetSelectRouting   -- SKNet's original "select" step (SKNet paper, Sec. 3.2):
        U   = sum_k F_k                     (fuse by SUMMATION first)
        s   = GAP(U)                        (squeeze the SUMMED feature)
        z   = ReLU(W_shared s)              (shared squeeze to d = max(C/16, 32))
        a_k = W_k z                         (branch-specific linear maps)
        w   = softmax_k(a) per channel      (select)
        out = sum_k w_k * F_k
      Note: this is exactly the structure Reviewer 4 referred to.

  V2  ACSSRChannelRouting  -- the ACSSR described by Eqs. (5)-(8) of the paper:
        v   = Concat(GAP(F_1..F_K))         (per-scale descriptors, K*C dims)
        a   = W2( GELU( BN( W1 v ) ) )      (joint bottleneck MLP)
        A   = reshape to (K, C)
        w   = softmax over scale, per channel
        out = sum_k w_k * F_k
      Differences vs V1: descriptors per scale (identity-preserving) instead
      of one descriptor of the summed feature; one joint MLP over all
      (scale, channel) pairs instead of per-branch linear maps; BN+GELU.

  V3  ACSSRPixelRouting    -- the ACSSR as implemented in the released code
      (main.py AdaptiveCrossScaleSoftRouting): a 1x1-conv router runs on the
      full-resolution concatenated feature map, producing weights
      (B, K, C, H, W) -> pixel-wise AND channel-wise routing (spatially
      adaptive). This is STRICTLY more expressive (and heavier) than V2;
      comparing V2 vs V3 quantifies what the paper text vs the code actually
      compute, and whether pixel-wise routing is worth its cost.

Use count_params() to report the routing-module parameter cost per variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# V1: SKNet original select step (channel-shared descriptor of the sum)
# ----------------------------------------------------------------------
class SKNetSelectRouting(nn.Module):
    def __init__(self, channels, num_scales=5, reduction=16, min_dim=32):
        super().__init__()
        self.num_scales = num_scales
        d = max(channels // reduction, min_dim)  # SKNet convention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_shared = nn.Linear(channels, d)
        self.relu = nn.ReLU(inplace=True)
        self.fcs_branch = nn.ModuleList(
            [nn.Linear(d, channels) for _ in range(num_scales)]
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        u = torch.stack(features, dim=0).sum(dim=0)      # fuse by summation
        s = self.gap(u).flatten(1)                       # (B, C) of the SUM
        z = self.relu(self.fc_shared(s))                 # (B, d)
        logits = torch.stack([fc(z) for fc in self.fcs_branch], dim=1)  # (B,K,C)
        w = F.softmax(logits, dim=1)                     # per-channel select
        stacked = torch.stack(features, dim=1)           # (B,K,C,H,W)
        return (stacked * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


# ----------------------------------------------------------------------
# V2: paper ACSSR (per-scale GAP descriptors -> joint MLP -> channel-wise
#     softmax over scales). Matches Eqs. (5)-(8).
# ----------------------------------------------------------------------
class ACSSRChannelRouting(nn.Module):
    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid = max(1, (channels * num_scales) // reduction)
        # 1x1-conv formulation of the MLP bottleneck (applied to 1x1 maps)
        self.router = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels * num_scales, kernel_size=1, bias=False),
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        gap_each = [F.adaptive_avg_pool2d(f, 1) for f in features]   # per scale
        v = torch.cat(gap_each, dim=1)                                # (B, K*C,1,1)
        a = self.router(v)                                            # (B, K*C,1,1)
        a = a.view(B, self.num_scales, C)
        w = F.softmax(a, dim=1)                                       # (B,K,C)
        stacked = torch.stack(features, dim=1)
        return (stacked * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


# ----------------------------------------------------------------------
# V3: pixel-wise ACSSR (identical to main.py AdaptiveCrossScaleSoftRouting)
# ----------------------------------------------------------------------
class ACSSRPixelRouting(nn.Module):
    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid = max(1, (channels * num_scales) // reduction)
        self.router = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels * num_scales, kernel_size=1, bias=False),
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        concat_f = torch.cat(features, dim=1)
        attn = self.router(concat_f)                     # (B, K*C, H, W)
        attn = attn.view(B, self.num_scales, C, H, W)
        attn = F.softmax(attn, dim=1)                    # pixel-wise select
        stacked = torch.stack(features, dim=1)
        return (stacked * attn).sum(dim=1)


# ----------------------------------------------------------------------
# V4: channel-level routing refined by per-pixel modulation (hybrid)
#     (identical to main.py AdaptiveCrossScaleSoftRouting after the 2026-10
#      revision: a GAP descriptor branch + a pixel branch scaled by beta)
# ----------------------------------------------------------------------
class ACSSRPixelChannelRouting(nn.Module):
    """Pixel-wise soft routing with a global (GAP) context term (current main.py)."""

    def __init__(self, channels, num_scales=5, reduction=4):
        super().__init__()
        self.num_scales = num_scales
        self.channels = channels
        mid = max(1, (channels * num_scales) // reduction)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(channels * num_scales, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, channels * num_scales, kernel_size=1, bias=False),
        )

    def forward(self, *features):
        B, C, H, W = features[0].shape
        concat_f = torch.cat(features, dim=1)
        attn = self.router(concat_f) + self.router(self.gap(concat_f))
        attn = attn.view(B, self.num_scales, C, H, W)
        attn = F.softmax(attn, dim=1)
        stacked = torch.stack(features, dim=1)
        return (stacked * attn).sum(dim=1)


# ----------------------------------------------------------------------
VARIANTS = {
    "sknet_select": SKNetSelectRouting,      # V1: baseline routing (SKNet)
    "acssr_channel": ACSSRChannelRouting,    # V2: paper equations (5)-(8)
    "acssr_pixel": ACSSRPixelRouting,        # V3: released-code version (before revision)
    "acssr_pixel_channel": ACSSRPixelChannelRouting,  # V4: hybrid (current main.py)
}


def build_fusion(name, channels=256, num_scales=5, reduction=4):
    if name not in VARIANTS:
        raise KeyError(f"unknown fusion variant '{name}', choose from {list(VARIANTS)}")
    if name == "sknet_select":
        return SKNetSelectRouting(channels, num_scales)
    return VARIANTS[name](channels, num_scales, reduction)


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    # ---------------- self-test ----------------
    torch.manual_seed(0)
    B, C, H, W, K = 2, 256, 64, 64, 5
    feats = [torch.randn(B, C, H, W) for _ in range(K)]

    print(f"{'variant':<16}{'params':>12}{'out shape':>22}")
    for name in VARIANTS:
        m = build_fusion(name, channels=C, num_scales=K)
        out = m(*feats)
        assert out.shape == (B, C, H, W), (name, out.shape)
        assert torch.isfinite(out).all(), name
        out.sum().backward()                       # gradient flows to all params
        grads_ok = all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in m.parameters()
        )
        assert grads_ok, name
        print(f"{name:<16}{count_params(m):>12,d}{str(tuple(out.shape)):>22}")

    # V2 weights must be constant over space (channel-wise); V3 must vary
    m2 = build_fusion("acssr_channel", C, K)
    with torch.no_grad():
        w2 = m2.router(torch.cat([F.adaptive_avg_pool2d(f, 1) for f in feats], 1))
    assert w2.shape[-2:] == (1, 1), "V2 routing operates on pooled 1x1 descriptors"
    print("V2 spatially constant (channel-wise)  -> OK (matches Eqs. 5-8)")

    m3 = build_fusion("acssr_pixel", C, K)
    with torch.no_grad():
        a3 = m3.router(torch.cat(feats, dim=1))
    assert a3.var(dim=(2, 3)).mean() > 1e-6, "V3 should vary over space"
    print("V3 spatially varying (pixel-wise)     -> OK (matches released code)")

    print("\nAll fusion-variant self-tests passed.")
