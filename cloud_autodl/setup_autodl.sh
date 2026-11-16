#!/bin/bash
# ============================================================
# AutoDL 环境配置脚本 — DACO/GETA 超参数实验
# ============================================================
# 使用方式:
#   1. 在 AutoDL 租用一台 GPU 实例 (推荐 RTX 3090 / RTX 4090)
#   2. 将项目代码上传到 /root/autodl-tmp/geta
#   3. 执行: bash autodl/setup_autodl.sh
# ============================================================
set -e

echo "========================================"
echo "  DACO/GETA AutoDL 环境配置"
echo "========================================"

# ---- 0. 路径配置 ----
PROJECT_DIR="/root/autodl-tmp/geta"
DATA_DIR="/root/autodl-tmp/data"
CONDA_ENV="daco"

cd "$PROJECT_DIR"
echo "[0] 项目目录: $PROJECT_DIR"

# ---- 1. 创建 conda 环境 ----
echo "[1] 创建 conda 环境: $CONDA_ENV ..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "    环境已存在，跳过创建"
else
    conda create -n "$CONDA_ENV" python=3.10 -y
fi

source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"

# ---- 2. 安装 PyTorch (CUDA 11.8 / 12.1 自适应) ----
echo "[2] 安装 PyTorch ..."
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
echo "    GPU 驱动版本: $CUDA_VERSION"

# AutoDL 镜像通常自带 CUDA，根据 torch 官方推荐安装
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# ---- 3. 安装其他依赖 ----
echo "[3] 安装 Python 依赖 ..."
pip install matplotlib numpy scipy pyyaml tqdm

# ---- 4. 下载 CIFAR-10 数据集 ----
echo "[4] 下载 CIFAR-10 数据集 ..."
mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/cifar-10-batches-py/readme.html" ] || [ -f "$DATA_DIR/cifar-10-python.tar.gz" ]; then
    echo "    CIFAR-10 已存在，跳过下载"
else
    cd "$DATA_DIR"
    # 方式1: torchvision 自动下载 (推荐)
    python -c "
import torchvision
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=True, download=True)
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=False, download=True)
print('CIFAR-10 下载完成')
"
    cd "$PROJECT_DIR"
fi

# ---- 5. 验证 GPU 可用性 ----
echo "[5] 验证 GPU ..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
    # 快速测试
    x = torch.randn(100, 100).cuda()
    print(f'  GPU 计算测试: OK (sum={x.sum().item():.2f})')
else:
    print('  [WARNING] CUDA 不可用！请检查 AutoDL 实例是否分配了 GPU')
"

# ---- 6. 验证 GETA 框架可导入 ----
echo "[6] 验证 GETA 框架 ..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model
from only_train_once.quantization.quant_layers import QuantizationMode
from sanity_check.backends.vgg7 import vgg7_bn
print('  GETA 框架导入: OK')
"

# ---- 7. 创建输出目录 ----
echo "[7] 创建输出目录 ..."
mkdir -p "$PROJECT_DIR/outputs/autodl_experiments"
mkdir -p "$PROJECT_DIR/outputs/autodl_experiments/checkpoints"
mkdir -p "$PROJECT_DIR/outputs/autodl_experiments/logs"
mkdir -p "$PROJECT_DIR/outputs/autodl_experiments/results"

echo ""
echo "========================================"
echo "  环境配置完成！"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 快速冒烟测试 (1 epoch 验证流程):"
echo "     python run_daco_cifar10.py --model vgg7 --epochs 1 --quick-test"
echo ""
echo "  2. 单个超参数实验:"
echo "     python run_daco_cifar10.py --model vgg7 --hparam beta_p --hvalue 0.1 --epochs 200"
echo ""
echo "  3. 批量跑全部超参数实验 (推荐 nohup):"
echo "     nohup bash autodl/run_autodl_experiments.sh > outputs/autodl_experiments/logs/batch.log 2>&1 &"
echo ""
echo "  4. 查看进度:"
echo "     tail -f outputs/autodl_experiments/logs/batch.log"
