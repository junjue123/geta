#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参数空间度量(E1 用): 相对 L2 / 余弦 Gram 距离 / 谱量。

为什么不是简单的 ‖ΔW‖₂
--------------------
ResNet 的 conv+BN 存在两组**严格保持函数不变**的重参数化自由度:
  (a) 输出通道置换 —— 交换 conv 的滤filters 顺序, 同步置换 BN 的 γ/β/μ/σ²;
  (b) 逐通道缩放 —— conv 第 c 个 filter 乘 s_c, BN 的 γ 除 s_c、μ 除 s_c、σ² 除 s_c²。
两者都不改变任何输入上的输出值, 但都会让 ‖ΔW‖₂ 大幅变化。因此「用 L2 衡量
两个模型的参数偏离」在原理上就不成立 —— 它把等价的重参数化当成了差异。

代价: 纯 SVD 谱熵**只对置换不变, 对逐通道缩放也不不变**。真正同时对 (a)(b)
不变的量是先做**行归一化**再取 Gram:
      R = Ŵ Ŵᵀ,   Ŵ 的每行是单位向量
R 是 filter 之间的余弦相似度矩阵。置换只会置换 R 的行列, 缩放被行归一化吸收,
所以 R 的谱量对两类变换都不变。这也正是 §10 里 "fold BN + 逐通道相对量" 的
严格版本。

约定
----
- 所有度量只在 conv 权重上算(BN 的可学习参数已被上面的重参数化吸收)。
- 被剪枝置零的 channel(整行权重全 0)无法定义方向, 计算时剔除并记录 valid 比例;
  两个模型共用**各自模型**的有效行掩码, 距离只在并集掩码上取,
  保证剪枝造成的"少一个通道"不会被误当成"方向改变"。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# 基础量
# ===========================================================================
def filters_of(w: torch.Tensor) -> torch.Tensor:
    """conv 权重 (out, in, kH, kW) -> (out, in*kH*kW)"""
    return w.detach().float().reshape(w.shape[0], -1)


def row_norm(w2d: torch.Tensor) -> torch.Tensor:
    return w2d.norm(dim=1)


def valid_mask(w2d: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """整行为 0 的 channel 视为已剪枝, 返回布尔掩码(True=有效)。"""
    return row_norm(w2d) > eps


def cosine_gram(w2d: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """行归一化后的 Gram 矩阵 R = Ŵ Ŵᵀ; 无效行置零。

    返回 (R, mask)。R 为 (out, out), 其对角在有效行上恒为 1。
    """
    if mask is None:
        mask = valid_mask(w2d)
    n = row_norm(w2d).clamp_min(1e-12)
    w_hat = w2d / n.unsqueeze(1)
    w_hat = torch.where(mask.unsqueeze(1), w_hat, torch.zeros_like(w_hat))
    return w_hat @ w_hat.t(), mask


def spectral_entropy(gram: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Gram 矩阵特征值的归一化谱熵 ∈[0,1], 只在有效行/列对应的子块上算。"""
    if mask is None:
        mask = torch.ones(gram.shape[0], dtype=torch.bool)
    g = gram[mask][:, mask]
    if g.numel() == 0:
        return float("nan")
    lam = torch.linalg.eigvalsh(g.double()).clamp_min(0)
    s = lam.sum()
    if s <= 0:
        return float("nan")
    p = lam / s
    p = p[p > 0]
    h = -(p * p.log()).sum().item()
    n_eff = int(mask.sum().item())
    return h / max(np.log(max(n_eff, 2)), 1e-12)


def stable_rank(gram: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Gram 的稳定秩 ‖·‖_F² / σ_max², 衡量有效自由度。"""
    if mask is None:
        mask = torch.ones(gram.shape[0], dtype=torch.bool)
    g = gram[mask][:, mask].double()
    if g.numel() == 0:
        return float("nan")
    fro2 = (g ** 2).sum()
    lmax = torch.linalg.eigvalsh(g).max().clamp_min(1e-12)
    return (fro2 / lmax).item()


def svd_spectrum(w2d: torch.Tensor) -> np.ndarray:
    """原始(未归一化)奇异值谱, 用作负对照。"""
    sv = torch.linalg.svdvals(w2d.double())
    return sv.numpy()


def gram_spectrum(gram: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """余弦 Gram 的特征值(降序)。这是**对置换与缩放都免疫**的量:

    置换 -> R 变为 P R Pᵀ, 特征值集合不变;
    逐通道缩放 -> 被行归一化吸收, R 本身不变。
    注意: 不能用 ‖R_a - R_b‖_F 比较 —— 置换会让 R 的行列同时换位,
    该量在函数完全相同时也会很大(实测 0.60)。
    """
    if mask is None:
        mask = torch.ones(gram.shape[0], dtype=torch.bool)
    g = gram[mask][:, mask].double()
    if g.numel() == 0:
        return torch.zeros(0, dtype=torch.float64)
    return torch.linalg.eigvalsh(g).flip(0).clamp_min(0)


def gram_spec_dist(ga: torch.Tensor, gb: torch.Tensor,
                   ma=None, mb=None) -> float:
    """两组余弦 Gram 的特征值向量的相对 L2 距离(置换+缩放不变)。"""
    la = gram_spectrum(ga, ma)
    lb = gram_spectrum(gb, mb)
    n = min(la.numel(), lb.numel())
    if n == 0:
        return float("nan")
    la, lb = la[:n], lb[:n]
    return ((la - lb).norm() / la.norm().clamp_min(1e-12)).item()


def svd_entropy(w2d: torch.Tensor) -> float:
    """纯 SVD 谱熵(负对照): 对置换不变, 对逐通道缩放**不**不变。"""
    sv = svd_spectrum(w2d)
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / max(np.log(max(len(sv), 2)), 1e-12))


# ===========================================================================
# 两模型之间的度量
# ===========================================================================
def conv_tensors(model_or_sd):
    """取出所有 conv 权重, 返回 OrderedDict{name: tensor}。"""
    if isinstance(model_or_sd, nn.Module):
        return {n: m.weight for n, m in model_or_sd.named_modules()
                if isinstance(m, nn.Conv2d)}
    out = {}
    for k, v in model_or_sd.items():
        if not torch.is_tensor(v) or v.dim() != 4:
            continue
        if k.rsplit(".", 1)[-1] != "weight":
            continue
        out[k] = v
    return out


def metrics_between(a, b) -> dict:
    """计算 a、b 两个模型(或 state_dict)之间的四类参数空间度量。

    返回 dict:
      rel_l2         : 逐 tensor ‖ΔW‖₂/‖W_a‖₂ 的均值(对重参数化不免疫)
      gram_spec_dist : 余弦 Gram 特征值向量的相对距离均值(**置换+缩放都免疫**, E1-a 判据)
      cos_gram_dist  : 余弦 Gram 的 Frobenius 距离(只免疫缩放; 仅适用于通道已对齐的
                       同源模型, 如同一训练轨迹上的两个快照)
      spec_ent_delta : 谱熵差的绝对值均值(免疫)
      stable_rank_delta
      svd_ent_delta  : 纯 SVD 谱熵差(负对照, 对缩放不免疫)
      n_tensors, valid_ratio
    """
    wa = conv_tensors(a)
    wb = conv_tensors(b)
    common = [k for k in wa if k in wb]
    if not common:
        raise RuntimeError("两个模型没有共同的 conv 权重名")

    l2s, gsds, cgs, sed, srd, svde, ratios = [], [], [], [], [], [], []
    for k in common:
        fa = filters_of(wa[k])
        fb = filters_of(wb[k])
        if fa.shape != fb.shape:
            raise RuntimeError(f"形状不一致 {k}: {tuple(fa.shape)} vs {tuple(fb.shape)}")

        ma = valid_mask(fa)
        mb = valid_mask(fb)
        m = ma | mb  # 并集: 剪枝丢掉的通道不参与(否则会被当成方向改变)

        # (1) 相对 L2
        l2s.append(((fa - fb).norm() / fa.norm().clamp_min(1e-12)).item())

        # (2) 余弦 Gram 特征值距离 —— 置换+缩放不变
        Ra, _ = cosine_gram(fa, m)
        Rb, _ = cosine_gram(fb, m)
        gsds.append(gram_spec_dist(Ra, Rb, m, m))

        # (3) 余弦 Gram 的 Frobenius 距离 —— 只对缩放不变
        cgs.append(((Ra - Rb).norm() / Ra.norm().clamp_min(1e-12)).item())

        # (4) 标量谱量
        sed.append(abs(spectral_entropy(Ra, m) - spectral_entropy(Rb, m)))
        srd.append(abs(stable_rank(Ra, m) - stable_rank(Rb, m)))

        # (5) 负对照
        svde.append(abs(svd_entropy(fa[ma]) - svd_entropy(fb[mb])))

        ratios.append(m.float().mean().item())

    return {
        "rel_l2": float(np.mean(l2s)),
        "gram_spec_dist": float(np.nanmean(gsds)),
        "cos_gram_dist": float(np.mean(cgs)),
        "spec_ent_delta": float(np.nanmean(sed)),
        "stable_rank_delta": float(np.nanmean(srd)),
        "svd_ent_delta": float(np.nanmean(svde)),
        "n_tensors": len(common),
        "valid_ratio": float(np.mean(ratios)),
        "per_tensor_l2": l2s,
        "per_tensor_gram_spec": gsds,
        "per_tensor_cos_gram": cgs,
    }


# ===========================================================================
# 自检
# ===========================================================================
def _selftest():
    torch.manual_seed(0)
    print("=" * 74)
    print("selftest: 两类真对称变换下, 各度量的响应")
    print("=" * 74)

    w = torch.randn(32, 16, 3, 3)  # 模拟一个 conv 权重

    # --- (a) 输出通道置换
    perm = torch.randperm(32)
    w_perm = w[perm].clone()

    # --- (b) 逐通道缩放
    s = torch.rand(32) * 3 + 0.1
    w_scale = w * s.view(-1, 1, 1, 1)

    # --- 负对照: 打乱单个元素(非对称变换, 应当被所有度量察觉)
    w_noise = w.clone()
    w_noise[0, 0, 0, 0] += 0.5

    rows = []
    for name, w2 in [("permute", w_perm), ("scale", w_scale), ("noise", w_noise)]:
        d = metrics_between({"c.weight": w}, {"c.weight": w2})
        rows.append((name, d["rel_l2"], d["gram_spec_dist"], d["cos_gram_dist"],
                     d["spec_ent_delta"], d["svd_ent_delta"]))

    print(f"{'transform':<11}{'rel_l2':>9}{'gram_spec':>12}{'cos_gram_F':>12}"
          f"{'spec_ent':>11}{'svd_ent':>11}")
    for name, l2, gsd, cg, se, sve in rows:
        print(f"{name:<11}{l2:>9.4f}{gsd:>12.2e}{cg:>12.2e}{se:>11.2e}{sve:>11.2e}")

    print("\n判读:")
    print("  - permute / scale: 函数完全不变, 期望 gram_spec / spec_ent ≈ 0")
    print("  - permute: 期望 cos_gram_F 很大 —— 置换让 R 的行列同时换位, 不能用 ‖R‖_F 比")
    print("  - permute / scale: 期望 rel_l2 显著 > 0 (L2 把重参数化当成差异)")
    print("  - scale: 期望 svd_ent > 0 (纯 SVD 谱熵对缩放不免疫 -> 负对照成立)")
    print("  - noise: 期望所有度量都变大 (真实函数改变)")

    d_perm = metrics_between({"c.weight": w}, {"c.weight": w_perm})
    d_scale = metrics_between({"c.weight": w}, {"c.weight": w_scale})
    d_noise = metrics_between({"c.weight": w}, {"c.weight": w_noise})

    assert d_perm["gram_spec_dist"] < 1e-5, "置换破坏了 gram_spec 不变性"
    assert d_perm["spec_ent_delta"] < 1e-5, "置换破坏了谱熵不变性"
    assert d_perm["rel_l2"] > 1.0, "置换下 rel_l2 应当很大"
    assert d_perm["cos_gram_dist"] > 0.1, "置换下 cos_gram_F 应当很大(它不免疫置换)"
    assert d_scale["gram_spec_dist"] < 1e-5, "缩放破坏了 gram_spec 不变性"
    assert d_scale["cos_gram_dist"] < 1e-5, "缩放破坏了 cos_gram_F 不变性"
    assert d_scale["spec_ent_delta"] < 1e-5, "缩放破坏了谱熵不变性"
    assert d_scale["rel_l2"] > 0.3, "缩放下 rel_l2 应当明显"
    assert d_scale["svd_ent_delta"] > 1e-3, "纯 SVD 谱熵没有随缩放变化, 负对照不成立"

    # 敏感性判据用相对形式: 真实扰动应当比两类对称变换的残差高 3 个数量级以上
    residual = max(d_perm["gram_spec_dist"], d_scale["gram_spec_dist"])
    assert d_noise["gram_spec_dist"] > 1e-4, "真实扰动没有被 gram_spec 察觉"
    assert d_noise["gram_spec_dist"] > 1e3 * residual, \
        f"gram_spec 的信噪比不足: noise={d_noise['gram_spec_dist']:.2e} residual={residual:.2e}"
    assert d_noise["cos_gram_dist"] > 1e-3, "真实扰动没有被 cos_gram_F 察觉"

    print("\n[ok] 全部断言通过: gram_spec/谱熵对两类真对称变换免疫, 且对真实扰动敏感")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="参数空间度量 E1")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(_selftest())
    print(__doc__)
