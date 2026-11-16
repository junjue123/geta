#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""量化权重缓存的 A/B 提速测量。

做法: 同一个量化模型、同一批数据, 交替跑"开缓存 / 关缓存"两组(各 N 步),
交替是为了让后台训练带来的 GPU 竞争对两组影响一致。
输出 ms/step 与省下的毫秒数。
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


def clear_cache(model):
    for m in model.modules():
        m.__dict__.pop("_wq_cache", None)


def main(n_steps=12, batch=128):
    import train as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = T.build_model("resnet20", dev)
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    def one_step(use_cache):
        if not use_cache:
            clear_cache(model)          # 强制每步重算: 等价于"没有缓存"
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
        QL.mark_quant_step()

    # 预热(含 CUDA 初始化)
    for _ in range(3):
        one_step(True)
    torch.cuda.synchronize()

    res = {}
    for name, flag in (("on", True), ("off", False)):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            one_step(flag)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t0) * 1000.0 / n_steps

    print(f"batch={batch} steps={n_steps} device={dev}")
    print(f"  权重缓存 开: {res['on']:7.1f} ms/step")
    print(f"  权重缓存 关: {res['off']:7.1f} ms/step")
    saved = res["off"] - res["on"]
    print(f"  省下       : {saved:7.1f} ms/step  ({saved / res['off'] * 100:+.1f}%)")


if __name__ == "__main__":
    main()
