"""第3章: 教师池(T_hist / T_cf)。

两类教师的分工是固定的, 不要混用:
  T_hist —— 压缩轨迹上"还比较可靠"的历史快照(热身前段)。负责软标签、
            特征锚、ref 位。它可靠, 所以可以做"标准答案"。
  T_cf   —— 当前模型的冻结影子副本。它**不可靠**(和学生在同一压缩点上),
            所以只做逐视图的密集认可度奖励, 不做软标签、不做特征锚。

设计约束(踩过的坑):
  - 教师必须整份 eval + requires_grad=False, 否则 BN 的 running stats 会被
    学生的 batch 污染, 且反传会把梯度灌进教师参数。
  - checkpoint 可能是续训格式 {'model':...} 也可能是最终格式
    {'model_state_dict':...}, 两种都要能吃。加载后要核对 key 是否对得上,
    对不上直接报错而不是静默半加载。
"""

import copy
import glob
import os
import re

import torch


_CKPT_EP = re.compile(r"_ep(\d+)\.pt$")


def discover_hist_candidates(out_dir, model_name, mode):
    """扫描 {out}/checkpoints 下按 epoch 命名的快照, 返回 [(epoch, path), ...]。"""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    pat = os.path.join(ckpt_dir, f"{model_name}_{mode}_ep*.pt")
    out = []
    for p in glob.glob(pat):
        m = _CKPT_EP.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def _extract_state(obj):
    """兼容三种保存格式, 返回纯 model state dict。"""
    if isinstance(obj, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # 本身就是 state dict(键里带 "." 或以 weight/bias 结尾)
        if any("." in str(k) for k in obj.keys()):
            return obj
    raise ValueError("checkpoint 里找不到模型权重(model / model_state_dict)")


def load_teacher(model_factory, path, device):
    """用 model_factory 新建一份模型, 灌入 ckpt 权重, 冻结后返回。"""
    teacher = model_factory()
    teacher.to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = _extract_state(ckpt)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    # 量化模型的 state dict 里带 d_quant_wt/t_quant_wt 这类派生量, 缺几个不算错;
    # 但原生参数缺了就是加载错了, 必须报出来。
    fatal = [k for k in missing if "quant" not in k and "scale" not in k]
    if fatal:
        raise RuntimeError(f"{os.path.basename(path)} 加载失败, 缺关键参数: {fatal[:6]}")
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


class TeacherPool:
    """按开关组装教师集合; 缺资源时降级并记录原因, 不静默。"""

    def __init__(self, cfg, model, out_dir, model_name, mode, device,
                 model_factory, log=None):
        self.cfg = cfg
        self.device = device
        self.hist = []          # [(epoch, model)]
        self.cf = None
        self.notes = []
        self._log = log or (lambda tag, msg: None)

        t = cfg["teacher"]
        self._build_hist(t["hist"], out_dir, model_name, mode, model_factory)
        self._build_cf(t["cf"], model)

    # ---------------- T_hist ----------------
    def _build_hist(self, hcfg, out_dir, model_name, mode, model_factory):
        if not hcfg["enabled"]:
            return
        cands = discover_hist_candidates(out_dir, model_name, mode)
        if not cands:
            self.notes.append(
                f"T_hist 未建成: {out_dir}/checkpoints 下没有 {model_name}_{mode}_ep*.pt; "
                "依赖它的特征层与 ref 位本次运行会自动关闭")
            self._log("DISTILL", self.notes[-1])
            return

        k = max(1, int(hcfg["max_count"]))
        picked = self._select(cands, k, hcfg["require_warmup_mid"])
        for ep, path in picked:
            try:
                m = load_teacher(model_factory, path, self.device)
                self.hist.append((ep, m))
            except Exception as e:                      # 单个坏 ckpt 不该拖垮整次训练
                self._log("DISTILL", f"教师 ep{ep} 加载失败, 跳过: {e}")
        if self.hist:
            eps = [e for e, _ in self.hist]
            self.notes.append(f"T_hist 就绪: {len(self.hist)} 个(ep={eps})")
            self._log("DISTILL", self.notes[-1])

    @staticmethod
    def _select(cands, k, require_mid):
        """选"压缩最少、最可靠"的几个: 优先最早 epoch; 需要中段时再补一个中位。"""
        eps = [e for e, _ in cands]
        chosen = [cands[0]]
        if k >= 2 and require_mid and len(cands) > 1:
            mid = cands[len(cands) // 2]
            if mid[0] != chosen[0][0]:
                chosen.append(mid)
        for c in cands[1:]:
            if len(chosen) >= k:
                break
            if c[0] not in [x[0] for x in chosen]:
                chosen.append(c)
        return chosen[:k]

    # ---------------- T_cf ----------------
    def _build_cf(self, ccfg, model):
        """S1 影子副本: 结构、剪枝掩码、量化参数全都照抄当前模型。"""
        if not ccfg["enabled"]:
            return
        self.cf = copy.deepcopy(model)
        self.cf.requires_grad_(False)
        self.cf.eval()
        if ccfg["implement"] != "shadow":
            self.notes.append(f"teacher.cf.implement={ccfg['implement']} 尚未实现, "
                              "本次按 shadow 处理")
            self._log("DISTILL", self.notes[-1])

    # ---------------- 查询 ----------------
    @property
    def has_hist(self):
        return len(self.hist) > 0

    @property
    def has_cf(self):
        return self.cf is not None

    def primary_hist(self):
        """软标签/特征锚用第一个(最早 = 压缩最少 = 最可靠)。"""
        return self.hist[0][1] if self.hist else None

    def describe(self):
        return " ".join(self.notes) if self.notes else "教师池为空"
