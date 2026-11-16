#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化权重 step 内缓存: 正确性测试(CPU 即可)。

要验证三件事:
  1. 同一 token 内重复前向, 输出逐元素一致(缓存不能改变结果);
  2. 重复前向的梯度要照常累加到同一份量化参数上(不能因为复用图而丢梯度);
  3. 一旦 mark_quant_step() 且权重被优化器改过, 前向必须用新权重重算(不能吃到过期缓存)。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.quantization import quant_layers as QL


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8 * 8 * 8, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(x.reshape(x.shape[0], -1))


def main():
    from only_train_once.quantization.quant_model import model_to_quantize_model
    from only_train_once.quantization.quant_layers import QuantizationMode

    torch.manual_seed(0)
    model = model_to_quantize_model(Tiny(), quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
    model.train()
    x = torch.randn(2, 3, 8, 8)

    print("T1 同一 step 内重复前向: 输出必须一致")
    QL.mark_quant_step()
    y1 = model(x)
    y2 = model(x)
    assert torch.equal(y1, y2), "缓存改变了前向结果"
    print("  [ok] 两次前向逐元素一致")

    print("T2 重复前向的梯度必须累加到量化参数上")
    for p in model.parameters():
        p.grad = None
    y1.sum().backward()
    g_single = {n: (p.grad.detach().clone() if p.grad is not None else None)
                for n, p in model.named_parameters()}
    for p in model.parameters():
        p.grad = None
    QL.mark_quant_step()
    y1b = model(x)
    y2b = model(x)
    (y1b.sum() + y2b.sum()).backward()
    worst = 0.0
    for n, p in model.named_parameters():
        if g_single[n] is None:
            continue
        expect = g_single[n] * 2.0
        worst = max(worst, float((p.grad - expect).abs().max()))
    assert worst < 1e-5, f"梯度没有正确累加, 最大偏差 {worst}"
    print(f"  [ok] 双前向梯度 = 单前向 × 2 (最大偏差 {worst:.2e})")

    print("T3 换 step 且权重被改过: 必须用新权重重算, 不能吃过期缓存")
    with torch.no_grad():
        model.conv.weight.add_(0.5)
    QL.mark_quant_step()
    y_new = model(x)
    assert not torch.allclose(y_new, y2b), "权重改了, 输出却没变 -> 缓存过期"
    # 对照: 同一份新权重下, 关掉缓存重算应与 y_new 一致
    cached = y_new.clone()
    for m in model.modules():
        if hasattr(m, "_wq_cache"):
            del m._wq_cache
    y_fresh = model(x)
    assert torch.equal(cached, y_fresh), "缓存结果与实时重算不一致"
    print("  [ok] 权重变化后结果更新, 且与实时重算逐元素一致")

    print("\n[ok] 量化权重缓存测试全部通过")


if __name__ == "__main__":
    main()
