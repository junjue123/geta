#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""蒸馏开关组的测试: 默认值 / 未知键 / 枚举 / 跨字段依赖 / 覆盖 / 摘要。

每个断言对应一类真实会犯的错, 不通过就说明配置层不可信。
用法: python scripts/test_distill_config.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.distillation import config as DC


def expect_error(fn, tag, keyword=None):
    try:
        fn()
    except DC.DistillConfigError as e:
        if keyword and keyword not in str(e):
            raise AssertionError(f"[{tag}] 报错信息里没有 {keyword!r}: {e}")
        print(f"  [ok] {tag}: {str(e)[:90]}")
        return
    raise AssertionError(f"[{tag}] 期望报错, 但没有报")


def main():
    print("=" * 74)
    print("T1 默认值: 不加 distill 块时全关")
    cfg = DC.resolve(None)
    assert cfg["enabled"] is False
    assert cfg["teacher"]["hist"]["enabled"] is False
    assert cfg["decision"]["enabled"] is False
    assert cfg["feature"]["enabled"] is False
    assert cfg["weight"]["enabled"] is False
    assert DC.active_switches(cfg) == []
    print("  [ok] 默认全关, active=[]")

    print("\nT2 未知键必须报错(拼错键名不能被静默丢弃)")
    expect_error(
        lambda: DC.resolve({"teacher": {"hist": {"max_cnt": 3}}}),
        "未知键 max_cnt", keyword="max_cnt")

    print("\nT3 枚举校验")
    expect_error(
        lambda: DC.resolve({"feature": {"enabled": True, "divergence": "kl"},
                            "teacher": {"hist": {"enabled": True}}}),
        "非法枚举 kl", keyword="divergence")

    print("\nT4 跨字段依赖")
    expect_error(
        lambda: DC.resolve({"enabled": True, "feature": {"enabled": True}}),
        "特征层缺 T_hist", keyword="teacher.hist.enabled")
    expect_error(
        lambda: DC.resolve({"enabled": True,
                            "decision": {"enabled": True, "reward": {"teacher_agreement": 1.0}}}),
        "密集奖励缺 T_cf", keyword="teacher.cf.enabled")
    expect_error(
        lambda: DC.resolve({"enabled": True, "decision": {"enabled": True, "group_size": 1}}),
        "组大小=1", keyword="group_size")

    print("\nT5 合法组合应当通过(三个机制全开)")
    full = {
        "enabled": True,
        "teacher": {"hist": {"enabled": True},
                    "cf": {"enabled": True, "views": 3}},
        "decision": {"enabled": True, "group_size": 8,
                     "reward": {"teacher_agreement": 1.0, "clean_consistency": 0.5},
                     "advantage": {"normalize_std": False, "mode": "positive"},
                     "ref_kl": {"enabled": True, "beta": 0.03}},
        "feature": {"enabled": True, "taps": ["layer2", "layer3", "avg_pool"],
                    "divergence": "forward", "budget": "sensitivity"},
        "weight": {"enabled": True, "epsilon0": 0.01, "kappa": 1.0},
        "diagnostics": {"effective_rank": {"enabled": True, "p_star": 0.9}},
    }
    cfg = DC.resolve(full)
    act = DC.active_switches(cfg)
    assert len(act) == 6, act
    print(f"  [ok] 六项机制全开: {act}")

    print("\nT6 关掉机制但不改代码(缺失对照用)")
    import copy
    for off_path in ("teacher.cf.enabled", "feature.enabled", "weight.enabled"):
        c2 = copy.deepcopy(cfg)
        # 关 T_cf 时同步把依赖它的奖励项归零, 否则校验应当报错
        if off_path == "teacher.cf.enabled":
            c2["decision"]["reward"]["teacher_agreement"] = 0.0
        DC.apply_overrides(c2, [f"{off_path}=false"])
        assert off_path not in [f"{p}=true" for p in []]
        n_after = len(DC.active_switches(c2))
        assert n_after < len(act), f"关 {off_path} 后机制数没减少"
        print(f"  [ok] 关 {off_path} -> 机制数 {len(act)} -> {n_after}")

    print("\nT7 dotted 覆盖(命令行/脚本用), 且类型按默认值转换")
    c3 = copy.deepcopy(cfg)
    DC.apply_overrides(c3, ["decision.group_size=12", "weight.epsilon0=0.02",
                            "feature.divergence=reverse",
                            "decision.augment_modes=crop,flip,identity"])
    assert c3["decision"]["group_size"] == 12 and isinstance(c3["decision"]["group_size"], int)
    assert abs(c3["weight"]["epsilon0"] - 0.02) < 1e-12
    assert c3["feature"]["divergence"] == "reverse"
    assert c3["decision"]["augment_modes"] == ["crop", "flip", "identity"]
    print("  [ok] 覆盖生效, int/float/枚举/list 都按默认类型转换")

    expect_error(lambda: DC.apply_overrides(c3, ["decision.grp_size=8"]),
                 "覆盖路径拼错", keyword="不存在")

    print("\nT8 augment_modes 必须含 identity(干净视图是 reward 的第三项前提)")
    expect_error(lambda: DC.resolve({"enabled": True,
                                     "decision": {"enabled": True,
                                                  "augment_modes": ["crop", "flip"]}}),
                 "缺 identity", keyword="identity")

    print("\nT9 摘要输出可读(将直接进 train.log)")
    for ln in DC.summary_lines(cfg):
        print("   " + ln)
    assert DC.summary_lines(DC.resolve(None))[0].startswith("distill: OFF")

    print("\nT10 yaml 文件路径可用(没有 distill 块时等同全关)")
    p = ROOT / "configs" / "compare_resnet20.yaml"
    c4 = DC.load_from_yaml(p)
    assert c4["enabled"] is False
    print(f"  [ok] {p.name} 没有 distill 块 -> 全关, 不影响现有训练")

    print("\nT11 参考配置 distill_example.yaml 能被解析成六项机制全开")
    p2 = ROOT / "configs" / "distill_example.yaml"
    c5 = DC.load_from_yaml(p2)
    assert c5["enabled"] is True
    assert len(DC.active_switches(c5)) == 6, DC.active_switches(c5)
    print(f"  [ok] {p2.name} -> {len(DC.active_switches(c5))} 项机制全开")

    print("\nT12 train.py 已接入 distill 块(静态检查, 不启动训练)")
    src = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "DISTILL_CFG = None" in src, "train.py 缺少全局 DISTILL_CFG"
    assert '"distill": DISTILL_CFG' in src, "run_config.yaml 快照里没有 distill"
    assert "distillation import config" in src, "train.py 没有调用配置解析"
    print("  [ok] 全局变量 / 解析调用 / 快照写入 三处都在")

    print("\n[ok] 蒸馏开关组全部测试通过")


if __name__ == "__main__":
    main()
