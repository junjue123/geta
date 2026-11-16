#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CUDA Graph 预录制 A/B: 量化模型的前向+反向整体录成一张图, 每步只发一次 launch。

背景: profile 显示量化 step 有 10,781 个 kernel, 纯 launch-bound。CUDA Graph 正是
"预先 launch"——把整段 kernel 序列捕获一次, 之后 replay 只需一次提交。

只捕获 fwd+bwd(不含 opt.step), 因为:
  - 量化前向里没有 .item()(实测 0 次), 可以在图里;
  - opt.step() 里的 .item()(PyTorch foreach-Adam 每参数取标量)会打断捕获, 留在图外。

交替测量 eager / graph 两组, 让后台训练的 GPU 竞争对两边一致。
"""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.quantization import quant_layers as QL
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def main(batch=128, steps=12):
    import train as T
    dev = torch.device("cuda")
    torch.manual_seed(0)

    model, _ = T.build_model("resnet20", dev)
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(dev)
    model.train()

    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)
    crit = torch.nn.CrossEntropyLoss()

    # 静态度量用的占位张量: 图里所有输入输出必须是同一块显存
    static_x = torch.empty_like(x)
    static_y = torch.empty_like(y)

    def eager_fwd_bwd():
        loss = crit(model(x), y)
        loss.backward()

    for _ in range(4):
        model.zero_grad(set_to_none=True)
        eager_fwd_bwd()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        eager_fwd_bwd()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) * 1000 / steps

    print(f"eager  fwd+bwd: {eager_ms:7.1f} ms/step")

    # ---- CUDA Graph 捕获 ----
    try:
        g = torch.cuda.CUDAGraph()
        # 捕获前必须热身, 否则 cuBLAS/cuDNN 的初始化会在捕获里做, 直接失败
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                eager_fwd_bwd()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        static_x.copy_(x)
        static_y.copy_(y)
        with torch.cuda.graph(g):
            static_out = model(static_x)
            loss = crit(static_out, static_y)
            loss.backward()
        torch.cuda.synchronize()
        print("捕获成功")
    except Exception as exc:                     # noqa: BLE001
        print(f"捕获失败: {type(exc).__name__}: {str(exc)[:160]}")
        return

    def graph_fwd_bwd():
        static_x.copy_(x)
        static_y.copy_(y)
        g.replay()

    for _ in range(4):
        model.zero_grad(set_to_none=True)
        graph_fwd_bwd()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        graph_fwd_bwd()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) * 1000 / steps

    print(f"graph  fwd+bwd: {graph_ms:7.1f} ms/step  ({(eager_ms - graph_ms) / eager_ms * 100:+.1f}%)")

    # 正确性: 图重放出来的 logits 必须与 eager 一致(同输入同权重)
    with torch.no_grad():
        g.replay()
        ref = static_out.clone()
        eager_out = model(static_x)
    worst = float((ref - eager_out).abs().max())
    print(f"图内重放输出 vs eager 前向: 最大偏差 {worst:.2e}")
    print("注意: 图内 backward 写的是捕获时的静态 grad buffer, 不是 p.grad; "
          "真要用于训练需在 replay 后把 grad 拷回参数(或用 make_graphed_callables)。")


if __name__ == "__main__":
    main()
