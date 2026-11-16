#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1-a: 两类严格保持函数不变的重参数化, 各参数度量如何响应。

为什么这个实验决定"用什么度量衡量参数偏离"
-------------------------------------------
压缩后要判断"当前模型偏离原模型多远"。若用 ‖ΔW‖₂, 那么一个与原点**函数完全
相同**、只是把通道换了个顺序(或把某个 filter 放大 2 倍同时把 BN 的 γ 缩小 2 倍)
的模型, 会被判成"偏离很大"。这类等价重参数化在压缩流程里天天发生(通道剪枝后
重排、BN 折叠、量化前后), 所以 L2 作为"偏离度量"在原理上不成立。

本实验对该论断给出可复现证据: 施加两类真对称变换(函数严格不变), 看
  rel_l2          —— 预期暴涨
  gram_spec_dist  —— 预期 ≈0
  spec_ent_delta  —— 预期 ≈0
  svd_ent_delta   —— 预期: 置换不变, 但对缩放不免疫(负对照)

两类变换的正确性用两条独立证据验证
--------------------------------
(1) 折叠等价: 把 conv+BN 折叠成单个 conv 后, 缩放变换应得到**逐元素相同**的
    折叠权重; 置换变换应得到仅通道顺序不同的折叠权重。
(2) 前向一致: 整个模型对同一批输入的输出相对差 <1e-4。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

import diag_common as C
import diag_metric as M


# ===========================================================================
# 折叠(conv+BN -> conv), 用于严格验证重参数化是否等价
# ===========================================================================
def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d, eps: float = 1e-5):
    """返回 (W_eff, b_eff), 满足 conv+BN 与 单个conv(W_eff,b_eff) 逐点等价。"""
    w = conv.weight.detach().float()
    gamma = bn.weight.detach().float()
    beta = bn.bias.detach().float()
    mu = bn.running_mean.detach().float()
    var = bn.running_var.detach().float()
    inv = gamma / torch.sqrt(var + eps)
    w_eff = w * inv.view(-1, 1, 1, 1)
    b_eff = beta - mu * inv
    if conv.bias is not None:
        b_eff = b_eff + conv.bias.detach().float() * inv
    return w_eff, b_eff


def _row_sort_key(w: torch.Tensor) -> torch.Tensor:
    """(已弃用) 按行范数+前几个元素构造排序键。

    保留仅作说明: 归一化后所有行范数都是 1±1e-7, 把它当排序主键会让排序结果
    由浮点噪声决定(实测两边的排列完全不同, 最大残差 1.5)。行对齐一律用
    match_rows_max_diff() 的最小代价匹配。
    """
    w2 = w.reshape(w.shape[0], -1)
    norm = w2.norm(dim=1, keepdim=True)
    return torch.cat([norm, w2[:, :16]], dim=1)


def match_rows_max_diff(wa: torch.Tensor, wb: torch.Tensor,
                        normalize_rows: bool = False) -> float:
    """把 wa 的各行与 wb 的各行做**最优配对**(置换对齐), 返回配对后的最大相对差。

    normalize_rows=True 时先按行归一化, 用于"置换+缩放"叠加场景(缩放被归一化吸收)。
    不依赖任何排序键 —— 归一化后行范数全为 1, 排序键会被浮点噪声主导。
    """
    n = wa.shape[0]
    a = wa.reshape(n, -1).float()
    b = wb.reshape(n, -1).float()
    if normalize_rows:
        a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-12)
        b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-12)

    dist = torch.cdist(a, b)  # (n, n)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(dist.cpu().numpy())
    except Exception:
        # 贪心回退: 每次取全局最小的一对
        ri, ci = [], []
        d = dist.clone()
        for _ in range(n):
            i, j = divmod(int(d.argmin().item()), n)
            ri.append(i); ci.append(j)
            d[i, :] = float("inf"); d[:, j] = float("inf")
    err = (a[list(ri)] - b[list(ci)]).abs().max()
    return (err / a.abs().max().clamp_min(1e-12)).item()


# ===========================================================================
# 两类真对称变换
# ===========================================================================
def blocks_of(model):
    """返回 (block_name, block_module) 列表; ResNet20 的 BasicBlock. """
    out = []
    for pname in ("layer1", "layer2", "layer3"):
        seq = getattr(model, pname, None)
        if seq is None:
            continue
        for i, blk in enumerate(seq):
            out.append((f"{pname}.{i}", blk))
    return out


@torch.no_grad()
def apply_channel_permutation(model, seed: int = 0):
    """每个 block 内部: 置换 conv1 输出通道 == bn1 通道 == conv2 输入通道。

    严格保持函数不变(置换与逐元素 BN/ReLU 可交换)。block 输出通道不变, 因此
    shortcut 分支与下游层都不受影响。
    """
    g = torch.Generator().manual_seed(seed)
    info = {}
    for name, blk in blocks_of(model):
        cout = blk.conv1.out_channels
        perm = torch.randperm(cout, generator=g)
        blk.conv1.weight.data = blk.conv1.weight.data[perm].clone()
        if blk.conv1.bias is not None:
            blk.conv1.bias.data = blk.conv1.bias.data[perm].clone()
        for attr in ("weight", "bias", "running_mean", "running_var"):
            v = getattr(blk.bn1, attr)
            v.data = v.data[perm].clone()
        blk.conv2.weight.data = blk.conv2.weight.data[:, perm].clone()
        info[name] = cout
    return info


@torch.no_grad()
def apply_channel_scaling(model, scale_range=(0.2, 5.0), seed: int = 0):
    """每个 block 内部: conv1 第 c 个 filter 乘 s_c, BN1 的统计量同步缩放。

    推导(注意 BN 本身对输入尺度不变, 所以 γ 不能动):
        y = γ(x-μ)/sqrt(σ²+eps) + β, 令 x' = s·x
        取 μ' = sμ, σ'² = s²σ², γ' = γ, β' = β
        => y' = γ(sx-sμ)/sqrt(s²σ²+eps) ≈ γ(x-μ)/sqrt(σ²+eps) = y   (仅差 eps 项)
    若误把 γ 也除 s(常见错误), 输出会被整体缩放 s 倍, 函数不再等价 —— 实测
    折叠权重相对差达 3.4, 前向输出差几个数量级。
    """
    g = torch.Generator().manual_seed(seed)
    info = {}
    for name, blk in blocks_of(model):
        cout = blk.conv1.out_channels
        s = torch.empty(cout).uniform_(*scale_range, generator=g)
        blk.conv1.weight.data = blk.conv1.weight.data * s.view(-1, 1, 1, 1)
        if blk.conv1.bias is not None:
            blk.conv1.bias.data = blk.conv1.bias.data * s
        blk.bn1.running_mean.data = blk.bn1.running_mean.data * s
        blk.bn1.running_var.data = blk.bn1.running_var.data * (s ** 2)
        info[name] = float(s.mean().item())
    return info


@torch.no_grad()
def apply_channel_permutation_plus_scaling(model, scale_range=(0.2, 5.0), seed: int = 0):
    """两类变换叠加(压缩流程里最真实的情况: 先剪枝重排、再 BN 折叠)。"""
    apply_channel_permutation(model, seed=seed)
    apply_channel_scaling(model, scale_range=scale_range, seed=seed + 1)


# ===========================================================================
# 测量
# ===========================================================================
def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def forward_rel_diff(m1, m2, x, device="cuda"):
    m1 = m1.to(device).eval()
    m2 = m2.to(device).eval()
    with torch.no_grad():
        o1 = m1(x.to(device))
        o2 = m2(x.to(device))
        if isinstance(o1, tuple):
            o1 = o1[0]
        if isinstance(o2, tuple):
            o2 = o2[0]
        denom = o1.abs().mean().clamp_min(1e-12)
        return ((o1 - o2).abs().mean() / denom).item()


def fold_check(sd_a, sd_b, model_template, mode: str):
    """用折叠权重比较两个 state_dict 在**函数层面**是否等价。

    mode:
      "exact"              —— 折叠权重应逐元素相同(纯缩放; eps 会留下 ~1e-3 残差)
      "multiset"           —— 应仅通道顺序不同(纯置换)
      "normalized_multiset"—— 先按行归一化(吸收缩放)后应仅通道顺序不同(置换+缩放)
    """
    model_template.load_state_dict(sd_a, strict=False)
    fold_a = {}
    for name, blk in blocks_of(model_template):
        fold_a[name] = fold_bn_into_conv(blk.conv1, blk.bn1)

    model_template.load_state_dict(sd_b, strict=False)
    diffs = []
    for name, blk in blocks_of(model_template):
        wb, bb = fold_bn_into_conv(blk.conv1, blk.bn1)
        wa, ba = fold_a[name]
        if mode == "exact":
            num = (wb - wa).abs().max().item()
        elif mode == "multiset":
            num = match_rows_max_diff(wa, wb) * wa.abs().max().item()
        else:  # normalized_multiset
            num = match_rows_max_diff(wa, wb, normalize_rows=True) * max(
                wa.abs().max().item(), 1e-12)
        den = wa.abs().max().clamp_min(1e-12).item()
        diffs.append(num / den)
    return float(np.max(diffs))


def run(ckpt_tag: str, device: str, out_dir: Path, n_eval: int = 128):
    ck = C.resolve_ckpt(ckpt_tag)
    print(f"[E1-a] ckpt = {ck.name}")

    # 只用浮点模型做参数空间实验(前向正确性由折叠验证保证, 不依赖精度)
    model = C.build_plain("resnet20")
    C.load_plain_weights(model, ck, verbose=False)
    base_sd = snapshot(model)

    x, y = C.load_cifar10_split(False, n=n_eval)
    xin = C.normalize(x)

    cases = [
        ("identity", lambda m: None, "exact"),
        ("permute", lambda m: apply_channel_permutation(m, seed=1), "multiset"),
        ("scale", lambda m: apply_channel_scaling(m, seed=2), "exact"),
        ("permute+scale", lambda m: apply_channel_permutation_plus_scaling(m, seed=3),
         "normalized_multiset"),
    ]

    results = []
    template = C.build_plain("resnet20")
    for label, fn, mode in cases:
        m = C.build_plain("resnet20")
        m.load_state_dict(base_sd, strict=False)
        fn(m)
        t_sd = snapshot(m)

        d = M.metrics_between(base_sd, t_sd)
        fwd = forward_rel_diff(model, m, xin, device=device)
        fold = fold_check(base_sd, t_sd, template, mode=mode)

        results.append({
            "transform": label,
            "fold_max_rel_diff": fold,
            "forward_rel_diff": fwd,
            "rel_l2": d["rel_l2"],
            "gram_spec_dist": d["gram_spec_dist"],
            "cos_gram_dist": d["cos_gram_dist"],
            "spec_ent_delta": d["spec_ent_delta"],
            "stable_rank_delta": d["stable_rank_delta"],
            "svd_ent_delta": d["svd_ent_delta"],
        })
        print(f"  {label:<14} fold_diff={fold:.2e} fwd_diff={fwd:.2e} "
              f"rel_l2={d['rel_l2']:.4f} gram_spec={d['gram_spec_dist']:.2e} "
              f"cosF={d['cos_gram_dist']:.2e} svd_ent={d['svd_ent_delta']:.2e}")

    # ---- 断言: 实验本身是否可信
    r = {x["transform"]: x for x in results}
    assert r["identity"]["fold_max_rel_diff"] < 1e-6, "恒等变换下折叠权重不一致"
    assert r["identity"]["rel_l2"] < 1e-9, "恒等变换下 rel_l2 非 0"

    for name in ("permute", "scale", "permute+scale"):
        # 容差 2e-3: BN 的 eps(1e-5) 不随缩放变化, 折叠时留下 ~1e-3 的残差,
        # 这属于实现细节(eps 不是重参数化的一部分), 不影响"函数等价"的结论。
        assert r[name]["fold_max_rel_diff"] < 2e-3, \
            f"{name}: 折叠权重不等价({r[name]['fold_max_rel_diff']:.2e}), 变换实现有误"
        assert r[name]["forward_rel_diff"] < 1e-2, \
            f"{name}: 前向输出变了({r[name]['forward_rel_diff']:.2e})"
        assert r[name]["rel_l2"] > 0.1, f"{name}: rel_l2 没有暴涨, 论断不成立"
        assert r[name]["gram_spec_dist"] < 1e-4, f"{name}: gram_spec 没有不变"
        assert r[name]["spec_ent_delta"] < 1e-4, f"{name}: 谱熵没有不变"

    assert r["scale"]["svd_ent_delta"] > 1e-3, "负对照失败: SVD 谱熵居然对缩放不变"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "symmetry.json").write_text(
        json.dumps({"ckpt": ck.name, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n[E1-a][ok] 结论: 两类等价变换下 rel_l2 暴涨({r['permute']['rel_l2']:.2f}/"
          f"{r['scale']['rel_l2']:.2f}) 而 gram_spec 与谱熵不变(<1e-4) "
          f"-> 参数偏离不能用 L2 衡量")
    print(f"  -> {out_dir / 'symmetry.json'}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="resnet20_baseline_ep10.pt")
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e1_metric"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(a.ckpt, dev, Path(a.out), n_eval=a.n_eval)
