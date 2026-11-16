#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E1-b: 参数偏离度量能否预测精度损失?

E1-a 只证明了"演进说参数偏离不能用 L2"。E1-b 反过来问: 那用哪一个?
做法是对同一模型做**渐进通道剪枝**(按 filter 范数置零), 每个剪枝档位测:
  - 实际精度(量化模型, 唯一正确的评估路径)
  - 各参数度量相对**未剪枝基准**的距离
然后看距离与精度损失(或精度本身)的 Spearman 相关。

判据(写死在断言里):
  - 至少一个"免疫类"度量(gram_spec / 谱熵)的 |rho| ≥ 0.7 且不低于 rel_l2 太多;
  - 若 rel_l2 的 rho 反而最高, 说明在当前数据上 L2 也够用 —— 这也是有效结论,
    但需要在论文里说明它只在"未发生重参数化"时成立(E1-a 已证其原理缺陷)。

剪枝实现: 逐 block 内部通道(conv1 输出 == bn1 通道), 按 filter L2 排序置零,
同时把该通道的 BN γ/β 置零, 使该通道输出恒为 0(与第2章 fix_pruned_groups_as_zeros
的做法一致: 置零而非物理删除, 形状不变)。
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
from diag_e1_symmetry import blocks_of


# ===========================================================================
# 通道剪枝(置零式)
# ===========================================================================
@torch.no_grad()
def prune_internal_channels(model, ratio: float, rank: str = "magnitude",
                            seed: int = 0, stage: str = "all"):
    """按比例置零 conv1 输出通道(同步置零 bn1 的 γ/β)。

    stage: "all" | "layer1" | "layer2" | "layer3" —— 只剪该层级的 block。
    ratio 是**存活通道**的比例: 已被剪枝置零的通道不计入分母, 否则在已压缩过的
    ckpt 上会先把本来就为 0 的通道再"剪"一遍, 得到 rel_l2 恰好为 0 的假变体
    (实测 daco_final 在 layer3 剪 10%~50% 时精度完全不变)。
    """
    if ratio <= 0:
        return 0
    g = torch.Generator().manual_seed(seed)
    total_pruned = 0
    for name, blk in blocks_of(model):
        if stage != "all" and not name.startswith(stage + "."):
            continue
        cout = blk.conv1.weight.shape[0]
        score = blk.conv1.weight.detach().reshape(cout, -1).norm(dim=1)
        live = score > 0
        n_live = int(live.sum())
        if n_live == 0:
            continue
        k = int(round(n_live * ratio))
        if k <= 0:
            continue
        if rank == "magnitude":
            # 只在存活通道里挑最小的 k 个
            order = torch.argsort(score)
            idx = order[live[order]][:k]
        else:
            live_idx = torch.nonzero(live).flatten()
            perm = torch.randperm(live_idx.numel(), generator=g)[:k]
            idx = live_idx[perm]
        blk.conv1.weight.data[idx] = 0.0
        if blk.conv1.bias is not None:
            blk.conv1.bias.data[idx] = 0.0
        blk.bn1.weight.data[idx] = 0.0
        blk.bn1.bias.data[idx] = 0.0
        total_pruned += k
    return total_pruned


def count_zero_channels(model):
    tot = zero = 0
    for _, blk in blocks_of(model):
        c = blk.conv1.weight.shape[0]
        z = int((blk.conv1.weight.detach().reshape(c, -1).norm(dim=1) == 0).sum())
        zero += z; tot += c
    return zero, tot


# ===========================================================================
# Spearman(自己算, 避免额外依赖)
# ===========================================================================
def _rank(x):
    x = np.asarray(x, dtype=float)
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # 处理并列: 取平均秩
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ===========================================================================
# 主流程
# ===========================================================================
def run(ckpt_tags, ratios, rank, device, out_dir, n_eval):
    x, y = C.load_cifar10_split(False, n=n_eval)
    xin = C.normalize(x)

    per_ckpt = {}
    for tag in ckpt_tags:
        ck = C.resolve_ckpt(tag)
        if not ck.exists():
            print(f"[skip] {tag}")
            continue
        print(f"\n[E1-b] ckpt = {ck.name}")

        # 基准: 浮点(度量用) + 量化(精度用)
        base_plain = C.build_plain("resnet20")
        C.load_plain_weights(base_plain, ck, verbose=False)
        base_sd = {k: v.detach().clone() for k, v in base_plain.state_dict().items()}
        base_quant, _ = C.load_quant_model(ck, device=device, verbose=False)
        base_acc = C.evaluate(base_quant, xin, y, device=device)
        print(f"  base_acc={base_acc:.4f}")

        rows = []
        # 变体: 全层均匀 + 按层级单独剪(后者让各度量的排序真正分开,
        # 否则单调的 ratio 会让所有度量的秩完全相同 -> Spearman 无区分力)
        variants = [("all", 0.0)]
        for stage in ("all", "layer1", "layer2", "layer3"):
            for r in ratios:
                if r > 0:
                    variants.append((stage, r))

        for stage, r in variants:
            p = C.build_plain("resnet20")
            p.load_state_dict(base_sd, strict=False)
            prune_internal_channels(p, r, rank=rank, seed=7, stage=stage)
            psd = {k: v.detach().clone() for k, v in p.state_dict().items()}

            q, _ = C.load_quant_model(ck, device=device, verbose=False)
            prune_internal_channels(q, r, rank=rank, seed=7, stage=stage)
            acc = C.evaluate(q, xin, y, device=device)

            d = M.metrics_between(base_sd, psd)
            # 额外的"最坏层"聚合: 单层剪枝时它对定位敏感, 与均值聚合会给出不同排序
            d["rel_l2_max"] = float(np.max(d["per_tensor_l2"]))
            d["gram_spec_max"] = float(np.nanmax(d["per_tensor_gram_spec"]))
            z, tot = count_zero_channels(p)
            rows.append({
                "stage": stage,
                "ratio": r,
                "acc": acc,
                "acc_drop": base_acc - acc,
                "zero_ratio": z / max(tot, 1),
                "rel_l2": d["rel_l2"],
                "rel_l2_max": d["rel_l2_max"],
                "gram_spec_dist": d["gram_spec_dist"],
                "gram_spec_max": d["gram_spec_max"],
                "cos_gram_dist": d["cos_gram_dist"],
                "spec_ent_delta": d["spec_ent_delta"],
                "stable_rank_delta": d["stable_rank_delta"],
                "svd_ent_delta": d["svd_ent_delta"],
            })
            print(f"  stage={stage:<7} ratio={r:.2f} acc={acc:.4f} "
                  f"drop={base_acc - acc:+.4f} rel_l2={d['rel_l2']:.4f} "
                  f"gram_spec={d['gram_spec_dist']:.3e}")

        # 相关性: 距离 vs 精度损失(用 |drop| 保证单调方向一致)
        metrics = ["rel_l2", "rel_l2_max", "gram_spec_dist", "gram_spec_max",
                   "cos_gram_dist", "spec_ent_delta", "stable_rank_delta",
                   "svd_ent_delta"]
        drops = [abs(r["acc_drop"]) for r in rows]
        corr = {m: spearman([r[m] for r in rows], drops) for m in metrics}
        corr_acc = {m: spearman([r[m] for r in rows], [r["acc"] for r in rows])
                    for m in metrics}

        print("\n  Spearman(distance, |acc_drop|):")
        for m in metrics:
            print(f"    {m:<18} rho={corr[m]:+.3f}   (vs acc: {corr_acc[m]:+.3f})")
        print(f"  区分力: 最大差异 {max(corr.values()) - min(corr.values()):+.3f}"
              f" (若≈0 说明这组变体仍无法区分度量)")

        per_ckpt[ck.name] = {
            "base_acc": base_acc,
            "rows": rows,
            "spearman_drop": corr,
            "spearman_acc": corr_acc,
        }

    # ---- 断言: 至少要有能预测的度量
    ok = []
    for name, r in per_ckpt.items():
        best_immune = max(r["spearman_drop"]["gram_spec_dist"],
                          r["spearman_drop"]["gram_spec_max"],
                          r["spearman_drop"]["spec_ent_delta"])
        ok.append((name, r["spearman_drop"]["rel_l2"], best_immune))
    assert per_ckpt, "没有可用 ckpt"
    best = max(max(v[1], v[2]) for v in ok)
    assert abs(best) >= 0.7, f"所有度量都无法预测精度损失(best |rho|={best:.3f})"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predict.json").write_text(
        json.dumps(per_ckpt, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[E1-b][ok] 汇总:")
    for name, l2r, imr in ok:
        print(f"  {name}: rho(L2)={l2r:+.3f}  rho(免疫类最佳)={imr:+.3f}")
    print(f"  -> {out_dir / 'predict.json'}")
    return per_ckpt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=["resnet20_baseline_ep10.pt", "resnet20_daco_final.pt"])
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--rank", default="magnitude", choices=["magnitude", "random"])
    ap.add_argument("--n-eval", type=int, default=1000)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "e1_metric"))
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run(a.ckpt, a.ratios, a.rank, dev, Path(a.out), a.n_eval)
