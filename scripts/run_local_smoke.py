"""本地最小冒烟测试：验证 GETA 核心压缩流程端到端可跑（不训练、不画图）。

验证链路：
  原始模型 -> 量化模型 -> OTO 图分析 -> random_set_zero_groups -> construct_subnet
  -> full/compressed 模型 forward 输出一致

跑法（d2l 环境，torch 2.1.0+cu118）：
  cd E:/pythonProject/geta-main/geta-main_origin
  python run_local_smoke.py
"""
import os
import sys
import torch

# 让 sanity_check.backends 可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sanity_check"))

from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from backends.vgg7 import vgg7_bn

OUT_DIR = "./cache/smoke"
os.makedirs(OUT_DIR, exist_ok=True)

# 冒烟测试强制 CPU：验证流程正确性即可，autodl 上再用 GPU。
# （model_to_quantize_model 包装后新增量化层默认在 CPU，量化后再 .to(device)
#   也能跑 GPU，但 CPU 冒烟更稳、足够覆盖端到端链路。）
device = "cpu"
print(f"[smoke] device={device}  torch={torch.__version__}")

dummy_input = torch.rand(1, 3, 32, 32).to(device)

# 1. 构建量化模型
model = vgg7_bn().to(device)
q_model = model_to_quantize_model(model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION)
print("[smoke] quantized model built OK")

# 2. OTO 图分析（跳过 visualize，避免依赖 graphviz 可执行文件）
oto = OTO(q_model, dummy_input)
print("[smoke] OTO graph built OK")

# 3. 指标计算（不画图）
full_params = oto.compute_num_params(in_million=True)
full_bops = oto.compute_bops(in_million=True)["total"]
print(f"[smoke] full: params={full_params:.4f}M  bops={full_bops:.4f}M")

# 4. 随机置零 + 构建子网（核心压缩流程）
oto.random_set_zero_groups()
oto.construct_subnet(out_dir=OUT_DIR)
print("[smoke] construct_subnet OK")

# 5. 加载并验证 forward 输出一致
full_model = torch.load(oto.full_group_sparse_model_path, map_location=device)
compressed_model = torch.load(oto.compressed_model_path, map_location=device)
full_model.eval()
compressed_model.eval()

with torch.no_grad():
    out_full = full_model(dummy_input)
    out_comp = compressed_model(dummy_input)

oto_comp = OTO(compressed_model, dummy_input)
comp_params = oto_comp.compute_num_params(in_million=True)
comp_bops = oto_comp.compute_bops(in_million=True)["total"]
print(f"[smoke] comp: params={comp_params:.4f}M  bops={comp_bops:.4f}M")

max_diff = (out_full - out_comp).abs().max().item()
print(f"[smoke] forward max abs diff (full vs compressed) = {max_diff:.6e}")

ok = max_diff < 1e-3
print(f"[smoke] RESULT: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
