"""
convert_checkpoint.py
将旧命名(0,2,3,4)的 DPRASR_Net 权重转换为新命名(0,1,2,3)。

用法:
    python convert_checkpoint.py --in best_busi.pth --out best_busi_new.pth
    python convert_checkpoint.py --in best_busi.pth --out best_busi_new.pth --verify

说明:
    - 旧 -> 新 的前缀映射:
        rcab2.   -> rcab1.
        rcab3.   -> rcab2.
        rcab4.   -> rcab3.
        linear2. -> linear1.
        linear3. -> linear2.
        linear4. -> linear3.
        aux2.    -> aux1.
        aux3.    -> aux2.
        aux4.    -> aux3.
    - linear0 / linear_b / aux0 / final_conv / acssr / encoder 保持不变。
    - 采用“单次前缀匹配”方式重命名，避免顺序 replace 导致的二次撞名。
"""

import argparse
import os
import sys

import torch


# 旧前缀 -> 新前缀
# 顺序无所谓：remap_key 对每个 key 只匹配一次并返回。
PREFIX_MAP = {
    "rcab2.":   "rcab1.",
    "rcab3.":   "rcab2.",
    "rcab4.":   "rcab3.",
    "linear2.": "linear1.",
    "linear3.": "linear2.",
    "linear4.": "linear3.",
    "aux2.":    "aux1.",
    "aux3.":    "aux2.",
    "aux4.":    "aux3.",
}


def remap_key(key):
    """对单个 key 做一次前缀匹配并替换；不命中则原样返回。"""
    for old_prefix, new_prefix in PREFIX_MAP.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def convert(in_path, out_path, verify=False):
    if not os.path.exists(in_path):
        print(f"❌ 找不到输入权重: {in_path}")
        sys.exit(1)

    print(f"📥 读取旧权重: {in_path}")
    state = torch.load(in_path, map_location="cpu")

    # 兼容存成 {"state_dict": ..., ...} 的写法
    wrapper = None
    if isinstance(state, dict) and "state_dict" in state and not any(
        k.startswith(("rcab", "linear", "aux", "encoder", "acssr", "final_conv"))
        for k in state.keys()
    ):
        wrapper = state
        state = state["state_dict"]

    # 逐 key 重映射
    new_state = {}
    changed = []
    for k, v in state.items():
        nk = remap_key(k)
        new_state[nk] = v
        if nk != k:
            changed.append((k, nk))

    # 安全检查：不允许出现重复 key
    if len(new_state) != len(state):
        print("❌ 转换后出现重复 key，映射表有冲突，已中止。")
        sys.exit(1)

    print(f"共 {len(state)} 个参数，其中 {len(changed)} 个 key 被重命名：")
    for old, new in sorted(changed):
        print(f"  {old}  ->  {new}")

    # 可选：加载进新模型做 strict 校验
    if verify:
        ok = False
        try:
            # 尝试从同目录的 DPRASR_Net.py 导入模型类
            sys.path.insert(0, os.path.dirname(os.path.abspath(in_path)))
            sys.path.insert(0, os.getcwd())
            from DPRASR_Net import DPRASR_Net  # noqa
            model = DPRASR_Net(embed_dim=256)
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            if missing:
                print(f"⚠️ 新模型中缺失的 key: {missing}")
            if unexpected:
                print(f"⚠️ 权重中多余的 key: {unexpected}")
            if not missing and not unexpected:
                print("✅ verify 通过：新权重与新模型 state_dict 完全匹配。")
                ok = True
            else:
                print("❌ verify 失败：存在 missing / unexpected key，请检查映射表。")
        except Exception as e:
            print(f"⚠️ verify 跳过（无法导入模型类或加载失败）: {e}")
            print("   仅做 key 重命名，未做 strict 校验。")

    # 写出
    if wrapper is not None:
        wrapper["state_dict"] = new_state
        torch.save(wrapper, out_path)
    else:
        torch.save(new_state, out_path)
    print(f"💾 已保存转换后的权重到: {out_path}")
    return ok if verify else True


def main():
    parser = argparse.ArgumentParser(description="转换 DPRASR_Net 旧命名权重 -> 新命名")
    parser.add_argument("--in", dest="in_path", default="best_UDIAT.pth",
                        help="旧权重路径 (默认 best_busi.pth)")
    parser.add_argument("--out", dest="out_path", default="best_UDIAT.pth",
                        help="新权重输出路径 (默认 best_busi_new.pth)")
    parser.add_argument("--verify", action="store_true",
                        help="转换后加载进新模型做 strict 校验")
    args = parser.parse_args()
    convert(args.in_path, args.out_path, verify=args.verify)


if __name__ == "__main__":
    main()
