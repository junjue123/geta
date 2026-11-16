#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化的一个训练 step 到底花在哪: 按 CUDA kernel 时间排序。

不做猜测, 直接看 top kernel 的占比与调用次数。若前几名都是 elementwise(小张量,
每次几百微秒但调用上千次), 那就是 launch-bound, 优化方向是"减少 kernel 数";
若是 conv/GEMM 占大头, 那优化方向是算力/精度(channels_last / AMP)。
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.quantization import quant_layers as QL
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def main(batch=128, steps=3, top=14):
    import train as T
    dev = torch.device("cuda")
    model, _ = T.build_model("resnet20", dev)
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    for _ in range(3):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step(); QL.mark_quant_step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    # torch 2.1 的 kineto 后端要求同时开 CPU activity, 否则 "Legacy CUDA profiling" 断言失败
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            opt.zero_grad(); crit(model(x), y).backward(); opt.step(); QL.mark_quant_step()
        torch.cuda.synchronize()

    rows = []
    for e in prof.key_averages():
        us = float(getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0))
        if us <= 0:
            continue
        rows.append((us / steps, e.count / steps, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)

    print(f"batch={batch} steps={steps}  每 step CUDA 时间 {total/1000:.1f} ms, "
          f"kernel 数 {int(sum(r[1] for r in rows))}")
    print(f"{'ms/step':>8} {'占%':>6} {'次数':>7}  kernel")
    for us, cnt, key in rows[:top]:
        print(f"{us/1000:8.2f} {us/total*100:5.1f}% {int(cnt):7d}  {key[:78]}")


if __name__ == "__main__":
    main()
