#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""第3章蒸馏的 YAML 开关组: schema / 校验 / 覆盖 / 生效摘要。

设计原则
--------
1) **默认全关**。不加 distill 块时, 训练行为与现在完全一致(纯第2章 DACO)。
2) **一个总开关 + 三个机制开关 + 两个教师开关**, 每个都能独立关;
   关掉某个开关要能让其它部分仍然合法(便于做"缺失对照"而不改代码)。
3) **参数全在 yaml 里**, 代码里不出现魔法数。改超参不改代码。
4) 未知键直接报错(而不是静默忽略): 拼错键名是这类配置最常见的坑,
   `teacher.hist.max_cnt` 写错会被静默丢弃, 然后你以为在跑 2 个教师其实跑了 1 个。

开关层级
--------
distill
├── enabled                  总开关
├── teacher
│   ├── hist.enabled         T_hist(可靠教师, 做软标签/特征锚/ref)
│   └── cf.enabled           T_cf(反事实教师, 只做逐视图密集奖励)
├── decision.enabled         决策层: 视图组 + reward + GRPO + ref KL
├── feature.enabled          表征层: 中间层特征 KL(走梯度注入)
├── weight.enabled           参数层: post-step ε-球投影
└── diagnostics.effective_rank.enabled    I5 停止准则(只读, 不改训练)
"""

import copy
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ===========================================================================
# 默认值(全部为"关闭/最保守"取值)
# ===========================================================================
DISTILL_DEFAULTS = {
    "enabled": False,

    # ---------------- 教师集合 ----------------
    "teacher": {
        "source": "trajectory",          # trajectory | none
        "hist": {                        # 可靠教师: 压缩更少 + acc 达标
            "enabled": False,
            "max_count": 2,
            "require_warmup_mid": True,  # 必须含热身中段快照(IB 论断的可证伪点)
            "coordinate": "compression", # compression | epoch
            "admit_acc_delta": 0.02,     # acc >= acc(theta_t) - delta 才准入
            "weight": "ib",              # ib | uniform
        },
        "cf": {                          # 反事实教师: 更宽松压缩配置套在当前权重
            "enabled": False,
            "implement": "shadow",       # shadow | none(只留 T_hist)
            "views": 2,                  # 参与 reward 的代表视图数(干净+2)
        },
    },

    # ---------------- 决策层 ----------------
    "decision": {
        "enabled": False,
        "group_size": 8,                 # G: 每个样本的视图数
        "augment_modes": ["identity", "crop", "flip", "color", "cutout", "rotate"],
        "severity": "mid",               # light | mid | heavy
        "reward": {
            "correctness": True,         # r += 1[pred==y]
            "teacher_agreement": 1.0,    # lambda_T: T_cf 认可度(主密集项)
            "clean_consistency": 0.5,    # lambda_C: 与干净视图同判且都对
        },
        "advantage": {
            "normalize_std": False,      # False = Dr.GRPO 形式(去 std)
            "mode": "positive",          # positive(只留正优势) | signed
        },
        "ref_kl": {                      # 压缩后策略漂移的锚(GRPO 里唯一剩下的 ref 位)
            "enabled": False,
            "beta": 0.03,
            "views": "clean",            # clean | all
        },
    },

    # ---------------- 表征层 ----------------
    "feature": {
        "enabled": False,
        "taps": ["layer2", "layer3", "avg_pool"],   # 特征取点
        "pick": "fixed",                 # fixed | deviation(按偏离量选)
        "divergence": "forward",         # forward(FKL) | reverse(RKL)
        "budget": "sensitivity",         # uniform | sensitivity
        "alpha": 1.0,
    },

    # ---------------- 参数层 ----------------
    "weight": {
        "enabled": False,
        "epsilon0": 0.01,                # 基础半径
        "kappa": 1.0,                    # epsilon_lc = epsilon0 * (s_bar/s_lc)^kappa
        "schedule": "tighten",           # tighten(压缩事件收紧+余弦放开) | constant
        "anchor": "pre_compression",     # pre_compression | init
    },

    # ---------------- 压缩事件后的强度缓升 ----------------
    "ramp": {
        "enabled": False,                # 与其余机制一致: 默认全关
        "steps": 20,                     # 剪枝后 N 步内 0->1 线性升(实测事件处 KL 涨 3~6 倍)
    },

    # ---------------- 诊断(只读) ----------------
    "diagnostics": {
        "effective_rank": {              # I5: 用有效秩判压缩是否该停
            "enabled": False,
            "p_star": 0.9,               # r_eff 阈值
            "every_steps": 50,
        },
    },
}

# 允许的枚举值(校验用)
ENUMS = {
    "teacher.source": ["trajectory", "none"],
    "teacher.hist.coordinate": ["compression", "epoch"],
    "teacher.hist.weight": ["ib", "uniform"],
    "teacher.cf.implement": ["shadow", "none"],
    "decision.severity": ["light", "mid", "heavy"],
    "decision.advantage.mode": ["positive", "signed"],
    "decision.ref_kl.views": ["clean", "all"],
    "feature.pick": ["fixed", "deviation"],
    "feature.divergence": ["forward", "reverse"],
    "feature.budget": ["uniform", "sensitivity"],
    "weight.schedule": ["tighten", "constant"],
    "weight.anchor": ["pre_compression", "init"],
}

VALID_TAPS = ["layer1", "layer2", "layer3", "layer4", "avg_pool"]


# ===========================================================================
# 工具
# ===========================================================================
def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _unflatten(flat):
    root = {}
    for key, val in flat.items():
        cur = root
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return root


def _set_path(cfg, path, value):
    cur = cfg
    parts = path.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            raise KeyError(f"路径不存在: {path} (在 '{p}' 处断了)")
        cur = cur[p]
    if parts[-1] not in cur:
        raise KeyError(f"路径不存在: {path}")
    cur[parts[-1]] = value


def _get_path(cfg, path):
    cur = cfg
    for p in path.split("."):
        cur = cur[p]
    return cur


def _coerce(value, ref):
    """按默认值的类型把 yaml/命令行里的值转过去。"""
    if isinstance(ref, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(ref, int) and not isinstance(ref, bool):
        return int(value)
    if isinstance(ref, float):
        return float(value)
    if isinstance(ref, list):
        if isinstance(value, str):
            return [s.strip() for s in value.split(",") if s.strip()]
        return list(value)
    return value


# ===========================================================================
# 解析 / 校验
# ===========================================================================
class DistillConfigError(ValueError):
    pass


def _check_keys(raw, schema, prefix=""):
    """未知键 -> 报错并给出最接近的合法键; 缺失键由 defaults 补。"""
    for k, v in raw.items():
        if k.startswith("_"):
            continue            # 内部键(_warnings 之类)不参与校验, 保证旧快照可读回
        if k not in schema:
            legal = list(_flat(schema).keys())
            near = difflib.get_close_matches(f"{prefix}{k}", legal, n=3, cutoff=0.6)
            hint = f", 形近的合法键: {near}" if near else ""
            raise DistillConfigError(f"未知键 '{prefix}{k}'{hint}")
        if isinstance(schema[k], dict):
            if not isinstance(v, dict):
                raise DistillConfigError(f"'{prefix}{k}' 应当是块(dict), 收到 {type(v).__name__}")
            _check_keys(v, schema[k], prefix + k + ".")


def resolve(raw=None, strict=True):
    """把 yaml 里的 distill 块解析成"默认值 + 覆盖"的完整配置。

    strict=True 时未知键报错; strict=False 时忽略未知键(不建议)。
    """
    cfg = copy.deepcopy(DISTILL_DEFAULTS)
    if not raw:
        return cfg
    if strict:
        _check_keys(raw, DISTILL_DEFAULTS)
    for k, v in raw.items():
        if k.startswith("_"):
            continue            # 内部键(如 _warnings)忽略, 保证 yaml 快照可往返
        if k not in DISTILL_DEFAULTS:
            continue
        if isinstance(v, dict):
            cfg[k] = _merge(cfg[k], v)
        else:
            cfg[k] = _coerce(v, DISTILL_DEFAULTS[k])
    validate(cfg)
    return cfg


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = _coerce(v, out.get(k, v))
    return out


def validate(cfg):
    """跨字段一致性校验。所有断言都以"开关组合仍合法"为前提。"""
    flat = _flat(cfg)

    for path, allowed in ENUMS.items():
        if flat.get(path) not in allowed:
            raise DistillConfigError(f"{path}={flat.get(path)!r} 不合法, 可选 {allowed}")

    # 数值范围
    if not (0.0 <= flat["decision.reward.teacher_agreement"] <= 5.0):
        raise DistillConfigError("decision.reward.teacher_agreement 应在 [0, 5]")
    if not (0.0 <= flat["decision.reward.clean_consistency"] <= 5.0):
        raise DistillConfigError("decision.reward.clean_consistency 应在 [0, 5]")
    if not (0.0 <= flat["decision.ref_kl.beta"] <= 1.0):
        raise DistillConfigError("decision.ref_kl.beta 应在 [0, 1]")
    if not (0.0 < flat["weight.epsilon0"] <= 1.0):
        raise DistillConfigError("weight.epsilon0 应在 (0, 1]")
    if flat["weight.kappa"] < 0:
        raise DistillConfigError("weight.kappa 应 >= 0")
    if not (0 <= flat["ramp.steps"] <= 200):
        raise DistillConfigError("ramp.steps 应在 [0, 200]")
    if not (0.0 < flat["diagnostics.effective_rank.p_star"] < 1.0):
        raise DistillConfigError("diagnostics.effective_rank.p_star 应在 (0,1)")

    modes = flat["decision.augment_modes"]
    if not isinstance(modes, list) or len(modes) == 0:
        raise DistillConfigError("decision.augment_modes 不能为空")
    if "identity" not in modes:
        raise DistillConfigError("decision.augment_modes 必须包含 identity(干净视图)")

    # ---- 依赖关系: 只在"相关开关确实开着"时检查
    if flat["feature.enabled"] and not flat["teacher.hist.enabled"]:
        raise DistillConfigError(
            "feature.enabled=true 需要 teacher.hist.enabled=true: 特征锚只能来自可靠教师")
    if flat["decision.ref_kl.enabled"] and not flat["teacher.hist.enabled"]:
        raise DistillConfigError(
            "decision.ref_kl.enabled=true 需要 teacher.hist.enabled=true: ref 位用 T_hist")

    if flat["decision.enabled"]:
        if flat["decision.group_size"] < 2:
            raise DistillConfigError("decision.group_size 至少为 2, 否则组内无优势可言")
        if flat["decision.group_size"] > 32:
            raise DistillConfigError("decision.group_size 过大(>32), 单步成本不可接受")
        if flat["decision.reward.teacher_agreement"] > 0 and not flat["teacher.cf.enabled"]:
            raise DistillConfigError(
                "reward.teacher_agreement>0 需要 teacher.cf.enabled=true: "
                "该奖励项来自 T_cf 的逐视图认可度; 若要关 T_cf 请把该项置 0")

    if flat["feature.enabled"]:
        for t in flat["feature.taps"]:
            if t not in VALID_TAPS:
                raise DistillConfigError(f"feature.taps 含未知层 '{t}', 可选 {VALID_TAPS}")

    # ---- master switch: 子开关开着但总开关关着, 只提醒不报错(便于整块复制)
    # 注意: 只在 collect_warnings() 里提示, **不得往 cfg 里写**。
    # 早先把 "distill.enabled=false, 子开关不生效" 写进 cfg["_warnings"],
    # 结果 run_config.yaml 快照里多了一个非 schema 键, 再读回来会被
    # _check_keys() 判成"未知键" -> 配置无法往返。这是踩过的坑。
    return cfg


def collect_warnings(cfg):
    """返回"能跑但要注意"的提示列表(不修改 cfg)。"""
    out = []
    flat = _flat(cfg)
    if not cfg.get("enabled"):
        minors = [k for k, v in flat.items()
                  if k.endswith(".enabled") and v is True
                  and not k.startswith("diagnostics.")]
        if minors:
            out.append(f"distill.enabled=false, 以下子开关不会生效: {minors}")
    if (cfg["weight"]["enabled"] and cfg["weight"]["schedule"] == "tighten"
            and float(cfg["weight"]["kappa"]) <= 1.0 + 1e-9):
        out.append("weight.schedule=tighten 但 kappa<=1.0: "
                   "eps = eps0 * kappa^-progress 恒等于 eps0, 收紧实际没生效")
    if cfg["ramp"]["enabled"] and not cfg["weight"]["enabled"]:
        out.append("ramp 靠 weight 层的 w_skip 探测压缩事件, 但 weight.enabled=false "
                   "-> 探测不到事件, 缓升不会触发")
    return out


# ===========================================================================
# 覆盖(命令行 / 实验脚本)
# ===========================================================================
def apply_overrides(cfg, overrides):
    """按 'a.b.c=value' 覆盖; 类型由默认值推断。"""
    for expr in overrides or []:
        if "=" not in expr:
            raise DistillConfigError(f"覆盖项应为 'path=value', 收到 {expr!r}")
        path, value = expr.split("=", 1)
        path = path.strip()
        if path.startswith("distill."):
            path = path[len("distill."):]
        try:
            ref = _get_path(DISTILL_DEFAULTS, path)
        except KeyError:
            legal = list(_flat(DISTILL_DEFAULTS).keys())
            near = difflib.get_close_matches(path, legal, n=3, cutoff=0.6)
            raise DistillConfigError(f"覆盖路径不存在: {path}, 形近: {near}")
        _set_path(cfg, path, _coerce(value, ref))
    validate(cfg)
    return cfg


# ===========================================================================
# 摘要
# ===========================================================================
def active_switches(cfg):
    """返回实际生效的机制名(受 master switch 约束)。"""
    if not cfg.get("enabled"):
        return []
    out = []
    if cfg["teacher"]["hist"]["enabled"]:
        out.append("T_hist")
    if cfg["teacher"]["cf"]["enabled"]:
        out.append("T_cf")
    if cfg["decision"]["enabled"]:
        out.append("决策层(视图组+GRPO)")
    if cfg["feature"]["enabled"]:
        out.append("表征层(特征KL)")
    if cfg["weight"]["enabled"]:
        out.append("参数层(ε-球)")
    if cfg["diagnostics"]["effective_rank"]["enabled"]:
        out.append("I5(r_eff)")
    return out


def summary_lines(cfg):
    """给人看的生效摘要(可直接进 train.log)。"""
    lines = []
    if not cfg.get("enabled"):
        lines.append("distill: OFF (训练行为等同纯第2章 DACO)")
        return lines
    lines.append(f"distill: ON  生效机制 = {active_switches(cfg)}")
    t = cfg["teacher"]
    lines.append(
        f"  教师: T_hist={'on' if t['hist']['enabled'] else 'off'}"
        f"(K≤{t['hist']['max_count']}, {t['hist']['coordinate']}, "
        f"w={t['hist']['weight']}, mid={t['hist']['require_warmup_mid']}) "
        f"T_cf={'on' if t['cf']['enabled'] else 'off'}({t['cf']['implement']})")
    d = cfg["decision"]
    if d["enabled"]:
        r = d["reward"]
        def _onoff(v):
            return "on" if (v is True or v == 1) else ("off" if (v is False or v == 0) else v)
        lines.append(
            f"  决策: G={d['group_size']} severity={d['severity']} "
            f"reward=(1:{_onoff(r['correctness'])}, T:{r['teacher_agreement']}, "
            f"C:{r['clean_consistency']}) adv={d['advantage']['mode']}"
            f"(std={d['advantage']['normalize_std']}) "
            f"ref_KL={'on' if d['ref_kl']['enabled'] else 'off'}(β={d['ref_kl']['beta']})")
    f = cfg["feature"]
    if f["enabled"]:
        lines.append(f"  表征: taps={f['taps']} pick={f['pick']} "
                     f"{f['divergence']} budget={f['budget']} α={f['alpha']}")
    w = cfg["weight"]
    if w["enabled"]:
        lines.append(f"  参数: ε0={w['epsilon0']} κ={w['kappa']} "
                     f"{w['schedule']} anchor={w['anchor']}")
    r = cfg.get("ramp") or {}
    if r.get("enabled"):
        lines.append(f"  压缩事件后缓升: {r['steps']} 步")
    if cfg["diagnostics"]["effective_rank"]["enabled"]:
        e = cfg["diagnostics"]["effective_rank"]
        lines.append(f"  I5: p*={e['p_star']} every={e['every_steps']} steps")
    for wmsg in collect_warnings(cfg):
        lines.append(f"  [WARN] {wmsg}")
    return lines


def load_from_yaml(path, overrides=None):
    """从 yaml 文件读 distill 块(文件里没有该块则返回全关配置)。"""
    import yaml
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = resolve(raw.get("distill"), strict=True)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="查看 distill 开关组")
    ap.add_argument("--config", required=False)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    if a.config:
        cfg = load_from_yaml(a.config, a.set)
    else:
        cfg = resolve(None)
        if a.set:
            cfg = apply_overrides(cfg, a.set)
    print("\n".join(summary_lines(cfg)))
