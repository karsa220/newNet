# -*- coding: utf-8 -*-
"""profile_model.py — 统计 DPRASR_Net 的 Params / FLOPs（审稿要求 R1-W10）
放仓库根目录运行（PYTHONPATH 已含仓库根）：
    python profile_model.py
    python profile_model.py --size 256
依赖: thop（pip install thop）
注意: thop 只统计 conv/linear（学界惯例，各论文口径一致）;
      FLOPs(G) = 2 x MACs(G)。
"""
import argparse
import torch
from thop import profile
from DPRASRNet.main import DPRASR_Net


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256, help="input resolution (square)")
    args = ap.parse_args()

    model = DPRASR_Net(embed_dim=256).eval()
    inp = torch.randn(1, 3, args.size, args.size)

    macs, params = profile(model, inputs=(inp,), verbose=False)
    total = count_params(model)
    print(f"\nInput           : 1 x 3 x {args.size} x {args.size}")
    print(f"Params (total)  : {total/1e6:.2f} M")
    print(f"Params (thop)   : {params/1e6:.2f} M")
    print(f"MACs            : {macs/1e9:.2f} G   ->  FLOPs = {2*macs/1e9:.2f} G")

    # 顶层子模块参数分布（看 ACSSR / DPRCAB 占比，评估轻量化空间）
    print("\nPer-module params:")
    rows = []
    for name, child in model.named_children():
        n = count_params(child)
        if n:
            rows.append((n, name))
    for n, name in sorted(rows, reverse=True):
        print(f"  {name:<28s} {n/1e6:8.2f} M  ({100*n/total:5.1f}%)")


if __name__ == "__main__":
    main()
