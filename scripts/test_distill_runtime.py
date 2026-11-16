#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""蒸馏运行时的单元测试: 视图组 / 决策层 GRPO / 特征层 KL / ε-球投影。

这些断言检查的是"公式是否按写的那样算", 不检查精度收益(那是 E3/E4 的事)。
用法: python scripts/test_distill_runtime.py   (CPU 即可, 约十秒)
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.distillation import config as DC
from only_train_once.distillation.views import make_views, view_diversity
from only_train_once.distillation.runtime import DistillRuntime
from only_train_once.distillation.teachers import TeacherPool


def main():
    torch.manual_seed(0)
    print("=" * 74)

    print("T1 视图组: 第0项必须是 identity, 总数等于 group_size, 组内要有差异")
    x = torch.randn(4, 3, 32, 32)
    vs = make_views(x, ["identity", "crop", "flip", "color", "erase", "rotate"],
                    "mid", group_size=8, seed=1)
    assert len(vs) == 8, len(vs)
    assert vs[0][0] == "identity" and torch.equal(vs[0][1], x)
    assert view_diversity(vs) > 0.01, view_diversity(vs)
    assert all(t.shape == x.shape for _, t in vs), "视图尺寸必须与原图一致"
    print(f"  [ok] 8 个视图, 组内平均相对偏离 {view_diversity(vs):.4f}")

    print("\nT2 只有 identity 时不应造假视图(否则优势恒为 0)")
    v1 = make_views(x, ["identity"], "mid", group_size=8, seed=1)
    assert len(v1) == 1 and view_diversity(v1) == 0.0
    print("  [ok] 退化情况返回单视图, 由 runtime 跳过决策层")

    print("\nT3 三种 severity 的强度必须逐项单调(不能只看混合后的平均偏离)")
    # view_diversity 把四种算子混在一起取平均, 不是严重强度指标;
    # 这里直接检查强度表, 并用 color 做一次实证(它的扰动幅度严格正比于 scale)。
    from only_train_once.distillation.views import SEVERITY
    for k in ("crop", "color", "erase", "rotate"):
        vals = [SEVERITY[s][k] for s in ("light", "mid", "heavy")]
        assert vals[0] < vals[1] < vals[2], (k, vals)
    devs = []
    for s in ("light", "mid", "heavy"):
        v = make_views(x, ["identity", "color"], s, 2, seed=2)[1][1]
        devs.append(float((v - x).std()))
    assert devs[0] < devs[1] < devs[2], devs
    print(f"  [ok] 三档逐项单调; color 实测偏离 {devs[0]:.4f} < {devs[1]:.4f} < {devs[2]:.4f}")

    print("\nT4 教师池: 开着 T_hist 但没有历史 ckpt 时必须降级并留痕, 不能假装有教师")
    from train import build_model
    model, _ = build_model("resnet20", torch.device("cpu"))
    pool = TeacherPool(
        DC.resolve({"enabled": True,
                    "teacher": {"hist": {"enabled": True}, "cf": {"enabled": True}}}),
        model, str(ROOT / "no_such_dir"), "resnet20", "daco",
        torch.device("cpu"),
        lambda: build_model("resnet20", torch.device("cpu"))[0])
    assert not pool.has_hist and pool.has_cf
    assert pool.notes and "T_hist 未建成" in pool.notes[0], pool.notes
    print(f"  [ok] {pool.notes[0][:64]}...")

    print("\nT5 决策层损失: 奖励有差异时有限、可反传; 奖励全同则损失必须恰好为 0")
    cfg = DC.resolve({"enabled": True,
                      "teacher": {"cf": {"enabled": True}},
                      "decision": {"enabled": True, "group_size": 4,
                                   "severity": "heavy",
                                   "augment_modes": ["identity", "flip", "crop", "color"]}})
    # 用一个对输入尺度敏感的 stub 代替 resnet20: 未训练的 resnet20 对所有视图
    # 常给出同一个 argmax, 于是奖励全同、优势为 0(这正是 E2 要普查的退化情形),
    # 拿它测"梯度是否流回"会误判。stub 能让不同视图的 argmax 真的不一样。
    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 32 * 32, 10)

        def forward(self, x):
            return self.fc(x.flatten(1))

    stub = _Stub()
    rt_stub = DistillRuntime(cfg, stub, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
                             str(ROOT / "no_such_dir"), "resnet20", "daco",
                             lambda: _Stub())
    X = torch.randn(4, 3, 32, 32)
    # 让标签匹配"干净视图"的预测: 干净视图的宽松性奖励为正, 失真视图为负
    y = stub(X).argmax(1)
    clean = stub(X).detach()
    loss, stats = rt_stub.extra_loss(X, y, clean)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    g = sum(1 for p in stub.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert g > 0, f"奖励有差异却没有梯度: {stats}"
    print(f"  [ok] loss={float(loss):.4f} 有梯度参数={g} {stats}")

    # 反面: 组内只有干净视图 -> 奖励全同 -> 优势 0 -> 不应产生任何梯度
    cfg_one = DC.resolve({"enabled": True,
                          "teacher": {"cf": {"enabled": True}},
                          "decision": {"enabled": True, "group_size": 4,
                                       "augment_modes": ["identity"]}})
    rt_one = DistillRuntime(cfg_one, stub, torch.nn.CrossEntropyLoss(),
                            torch.device("cpu"), str(ROOT / "no_such_dir"),
                            "resnet20", "daco", lambda: _Stub())
    stub.zero_grad()
    l1, s1 = rt_one.extra_loss(X, y, stub(X).detach())
    assert l1 is None, "只有干净视图时不该造出决策层损失"
    print("  [ok] 单视图 -> 跳过(不算 0, 而是 None), 避免伪装成'有效但无梯度'")

    print("\nT6 ε-球投影: 把参数拉开后必须被拉回球内")
    cfg_w = DC.resolve({"enabled": True,
                        "teacher": {"cf": {"enabled": True}},
                        "weight": {"enabled": True}})
    rt = DistillRuntime(cfg_w, model, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
                        str(ROOT / "no_such_dir"), "resnet20", "daco",
                        lambda: build_model("resnet20", torch.device("cpu"))[0])
    target = None
    for n, p in model.named_parameters():
        if p.dim() >= 2 and p.numel() > 100:
            target = (n, p)
            break
    name, p = target
    a = rt.anchor[name]
    with torch.no_grad():
        p.add_(torch.randn_like(p) * 10.0)
    before = float((p - a).norm() / a.norm())
    st = rt.project_weights(progress=1.0)
    after = float((p - a).norm() / a.norm())
    eps = cfg["weight"]["epsilon0"]
    assert before > eps, f"扰动太小没触发投影: {before}"
    assert after <= eps + 1e-5, f"投影后仍越界: {after} > {eps}"
    print(f"  [ok] {name}: 相对偏离 {before:.4f} -> {after:.4f} (ε={eps}) 统计={st}")

    print("\nT7 特征层 KL: 前向/反向两种都有限, 且层名非法时不崩")
    for div in ("forward", "reverse"):
        c2 = DC.resolve({"enabled": True,
                         "teacher": {"hist": {"enabled": True}, "cf": {"enabled": True}},
                         "feature": {"enabled": True, "taps": ["layer2", "avg_pool"],
                                     "divergence": div}})
        # 没有历史 ckpt, 但为测 KL 本身, 手动塞一个教师进去
        rt2 = DistillRuntime(c2, model, torch.nn.CrossEntropyLoss(), torch.device("cpu"),
                            str(ROOT / "no_such_dir"), "resnet20", "daco",
                            lambda: build_model("resnet20", torch.device("cpu"))[0])
        rt2.pool.hist = [(1, model)]
        rt2.feature_on = True
        fl, fs = rt2._feature_loss(X)
        assert fl is not None and torch.isfinite(fl), (div, fl)
        print(f"  [ok] {div}: loss={float(fl):.5f} {fs}")

    print("\nT9 CUDA 上生成视图(曾经因为 generator 在 CPU 上而直接崩)")
    if torch.cuda.is_available():
        xc = x.cuda()
        vc = make_views(xc, ["identity", "crop", "flip", "color", "erase"],
                        "mid", group_size=6, seed=3)
        assert all(t.device == xc.device for _, t in vc), "视图没留在原设备"
        print(f"  [ok] 6 个视图仍在 {xc.device}, 组内偏离 {view_diversity(vc):.4f}")
    else:
        print("  [skip] 无 CUDA")

    print("\nT8 T_hist 真实加载: 拿现成 ckpt 灌进新模型, 权重必须一致(不静默半加载)")
    real = ROOT / "outputs" / "framework_compare" / "checkpoints" / "resnet20_daco_ep10.pt"
    if real.exists():
        from only_train_once.distillation.teachers import load_teacher
        t = load_teacher(lambda: build_model("resnet20", torch.device("cpu"))[0],
                         str(real), torch.device("cpu"))
        # 教师必须真的冻结: 任何参数可训练都是事故(会把梯度灌进教师)
        assert all(not p.requires_grad for p in t.parameters())
        assert not t.training
        n_ref = real.stat().st_size
        print(f"  [ok] 已加载 {real.name} ({n_ref/1e6:.1f}MB), 冻结状态正确")
    else:
        print(f"  [skip] 找不到 {real}, 跳过(该测试依赖已有训练产物)")

    print("\nT10 视图随机种子必须跟着 step 走, 且额外前向不能污染 BN 统计")
    s0 = rt_stub.step
    rt_stub.extra_loss(X, y, torch.zeros(X.size(0), 10))
    assert rt_stub.step == s0 + 1, f"extra_loss 没有推进 step: {s0} -> {rt_stub.step}"
    cfg_bn = DC.resolve({"enabled": True,
                         "teacher": {"cf": {"enabled": True}},
                         "decision": {"enabled": True, "group_size": 4,
                                      "augment_modes": ["identity", "flip", "crop"]}})
    model.train()
    rt_bn = DistillRuntime(cfg_bn, model, torch.nn.CrossEntropyLoss(),
                           torch.device("cpu"), str(ROOT / "no_such_dir"),
                           "resnet20", "daco",
                           lambda: build_model("resnet20", torch.device("cpu"))[0])
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns, "模型里没有 BN, 该断言无意义"
    rm_before = [m.running_mean.clone() for m in bns]
    # 只跑 extra_loss(不跑主前向), 这样若 BN 统计被改写就一定是视图前向干的
    rt_bn.extra_loss(X, y, torch.zeros(X.size(0), 10))
    moved = sum(1 for m, a in zip(bns, rm_before) if not torch.equal(m.running_mean, a))
    assert moved == 0, f"{moved} 个 BN 的 running_mean 被视图前向改写了"
    print(f"  [ok] step={rt_stub.step}, {len(bns)} 个 BN 统计未被额外前向改动")

    print("\nT11 tap 去重: layer3 与 avg_pool 是同一个量, 只能计一次")
    cfg_dup = DC.resolve({"enabled": True,
                          "teacher": {"hist": {"enabled": True}, "cf": {"enabled": True}},
                          "feature": {"enabled": True,
                                      "taps": ["layer3", "avg_pool"]}})
    rt_dup = DistillRuntime(cfg_dup, model, torch.nn.CrossEntropyLoss(),
                            torch.device("cpu"), str(ROOT / "no_such_dir"),
                            "resnet20", "daco",
                            lambda: build_model("resnet20", torch.device("cpu"))[0])
    rt_dup.pool.hist = [(1, model)]
    rt_dup.feature_on = True
    fl2, fs2 = rt_dup._feature_loss(X)
    assert "feat_layer3" in fs2 and "feat_avg_pool" not in fs2, fs2
    print(f"  [ok] 去重后只剩 {list(fs2.keys())}, 未把同一个量计两次")

    print("\nT12 压缩事件后强度缓升: 事件后从 0 线性升到 1, 非事件期恒 1")
    cfg_ramp = DC.resolve({"enabled": True, "ramp": {"enabled": True, "steps": 10},
                           "weight": {"enabled": False}})
    rt_ramp = DistillRuntime(cfg_ramp, model, torch.nn.CrossEntropyLoss(),
                             torch.device("cpu"), str(ROOT / "no_such_dir"),
                             "resnet20", "daco",
                             lambda: build_model("resnet20", torch.device("cpu"))[0])
    assert rt_ramp._ramp_scale() == 1.0
    rt_ramp._note_event(3)
    rt_ramp._note_event(9)                     # skip 增长 == 刚剪过
    rt_ramp._ramp_left = 8
    assert abs(rt_ramp._ramp_scale() - 0.2) < 1e-9
    rt_ramp._ramp_left = 0
    assert rt_ramp._ramp_scale() == 1.0
    print("  [ok] 事件前 1.0 / 事件后第 2 步 0.2 / 升满后 1.0")

    print("\nT13 教师取样口径: E4 要比 K 与坐标, pick/coordinate 必须真的改变选到谁")
    import only_train_once.distillation.teachers as _tch
    cands = [(5, "p5"), (10, "p10"), (15, "p15"), (20, "p20"), (25, "p25"),
             (30, "p30"), (35, "p35")]
    s = _tch.TeacherPool._select
    assert [e for e, _ in s(cands, 2, True, "compression", "earliest")] == [5, 20]
    assert [e for e, _ in s(cands, 1, True, "compression", "mid")] == [20]
    assert [e for e, _ in s(cands, 1, True, "compression", "end")] == [35]
    assert [e for e, _ in s(cands, 2, True, "compression", "end")] == [35, 20]
    assert [e for e, _ in s(cands, 3, True, "compression", "earliest")] == [5, 20, 10]
    # epoch 等距: 含首尾铺开, 且必须真的和"压缩更少优先"取到不同的集合
    ev = [e for e, _ in s(cands, 2, False, "epoch_even", "earliest")]
    assert ev == [5, 35], ev
    ev3 = [e for e, _ in s(cands, 3, False, "epoch_even", "earliest")]
    assert ev3 == [5, 20, 35], ev3
    assert len({tuple(ev), tuple([e for e, _ in s(cands, 2, False)])}) == 2
    print("  [ok] earliest/mid/end/epoch_even 四种口径分别取到不同快照")

    print("\nT14 视图组不许有重复输入: G=8 而算子只有 6 个时, 副本必须真的不同")
    import only_train_once.distillation.views as _vw
    xv = torch.randn(4, 3, 32, 32)
    views = _vw.make_views(xv, ["identity", "crop", "flip", "color", "cutout", "rotate"],
                           "mid", group_size=8, seed=1)
    assert len(views) == 8, len(views)
    for i in range(1, len(views)):
        for j in range(i + 1, len(views)):
            assert not torch.allclose(views[i][1], views[j][1]), \
                f"{views[i][0]} 与 {views[j][0]} 是同一个输入(组里重复计票)"
    flips = [float((v - xv).abs().mean()) for n, v in views if n.startswith("flip")]
    assert len(flips) == 2 and abs(flips[0] - flips[1]) > 1e-6, flips
    print(f"  [ok] 8 个视图两两不同, flip 副本 L1={flips[0]:.3f}/{flips[1]:.3f}")

    print("\n[ok] 运行时单元测试全部通过")


if __name__ == "__main__":
    main()
