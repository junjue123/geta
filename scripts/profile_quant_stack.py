#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""抓 aten::item / _to_copy 的 Python 调用栈, 定位每步的隐式同步与拷贝来源。"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from only_train_once.quantization import quant_layers as QL
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode


def main(batch=128):
    import train as T
    dev = torch.device("cuda")
    model, _ = T.build_model("resnet20", dev)
    model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(batch, 3, 32, 32, device=dev)
    y = torch.randint(0, 10, (batch,), device=dev)

    for _ in range(2):
        opt.zero_grad(); crit(model(x), y).backward(); opt.step(); QL.mark_quant_step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 with_stack=True, record_shapes=False) as prof:
        opt.zero_grad(); crit(model(x), y).backward(); opt.step(); QL.mark_quant_step()
        torch.cuda.synchronize()

    # key_averages(with_stack=True) 才能拿到 stack 列表
    for target in ("aten::item", "aten::_to_copy", "aten::result_type"):
        for e in prof.key_averages(group_by_stack_n=3):
            if e.key != target:
                continue
            print(f"\n=== {target}  count={e.count} ===")
            for st in (e.stack or [])[:3]:
                loc = str(getattr(st, "stack", st))
                # 只保留仓库内的帧, 去掉 torch 内部噪音
                for ln in loc.strip().splitlines():
                    if "geta-main_origin" in ln:
                        print("   ", ln.strip()[:150])


if __name__ == "__main__":
    main()
