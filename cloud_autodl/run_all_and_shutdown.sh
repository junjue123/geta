#!/bin/bash
# ============================================================
# DACO 超参数实验一键启动 (跑完自动关机)
# ============================================================
# 流程: 等CIFAR-10下载 → 冒烟测试 → β_p扫描 → φ_b扫描 → κ扫描 → 收集结果 → 关机
# 预估 RTX 4090D: 19次 × ~20min ≈ 6-7 小时
# ============================================================
set -e

PROJECT_DIR="/root/autodl-tmp/geta"
DATA_DIR="/root/autodl-tmp/data"
OUT_DIR="$PROJECT_DIR/outputs/autodl_experiments"
LOG_DIR="$OUT_DIR/logs"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" "$OUT_DIR/results" "$OUT_DIR/checkpoints"

echo "========================================"
echo "  DACO 超参数实验一键启动"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start: $(date)"
echo "========================================"

# ---- 0. 等待 CIFAR-10 下载完成 ----
echo ""
echo "[0/5] 等待 CIFAR-10 下载..."
WAIT=0
while [ $WAIT -lt 300 ]; do
    if python -c "
import torchvision
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=True)
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=False)
" 2>/dev/null; then
        echo "  CIFAR-10 就绪! (${WAIT}s)"
        break
    fi
    echo "  等待中... ${WAIT}s $(date +%H:%M:%S)"
    sleep 10
    WAIT=$((WAIT + 10))
done

if [ $WAIT -ge 300 ]; then
    echo "  [ERROR] CIFAR-10 下载超时，尝试重新下载..."
    python -c "
import torchvision
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=True, download=True)
torchvision.datasets.CIFAR10(root='$DATA_DIR', train=False, download=True)
print('重新下载完成')
"
fi

# ---- 1. 冒烟测试 ----
echo ""
echo "[1/5] 冒烟测试 (1 epoch, VGG-7)..."
echo "  Start: $(date)"
python run_daco_cifar10.py --model vgg7 --epochs 1 --quick-test \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" 2>&1 | tee "$LOG_DIR/smoke_test.log"
echo "  Done: $(date)"

# ---- 2. β_p 扫描 (7 values) ----
echo ""
echo "========================================"
echo "[2/5] RCAJS β_p 敏感性扫描 (7 values × 200 epochs)"
echo "  Start: $(date)"
echo "========================================"
python run_daco_cifar10.py \
    --model vgg7 --hparam beta_p --epochs 200 \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt \
    2>&1 | tee "$LOG_DIR/sweep_beta_p.log"
echo "  β_p Done: $(date)"

# ---- 3. φ_b 扫描 (5 values) ----
echo ""
echo "========================================"
echo "[3/5] RCAJS φ_b 敏感性扫描 (5 values × 200 epochs)"
echo "  Start: $(date)"
echo "========================================"
python run_daco_cifar10.py \
    --model vgg7 --hparam phi_b --epochs 200 \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt \
    2>&1 | tee "$LOG_DIR/sweep_phi_b.log"
echo "  φ_b Done: $(date)"

# ---- 4. κ 扫描 (7 values) ----
echo ""
echo "========================================"
echo "[4/5] MCSS κ 敏感性扫描 (7 values × 200 epochs)"
echo "  Start: $(date)"
echo "========================================"
python run_daco_cifar10.py \
    --model vgg7 --hparam kappa --epochs 200 \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt \
    2>&1 | tee "$LOG_DIR/sweep_kappa.log"
echo "  κ Done: $(date)"

# ---- 5. 收集结果 ----
echo ""
echo "[5/5] 收集结果..."
python collect_results.py --out "$OUT_DIR" 2>&1 | tee "$LOG_DIR/collect.log"

# ---- 6. 关机 ----
echo ""
echo "========================================"
echo "  全部实验完成!"
echo "  End: $(date)"
echo "  结果目录: $OUT_DIR/results/"
echo "  汇总CSV: $OUT_DIR/experiment_summary.csv"
echo "  3 秒后自动关机 (关机不计费, 数据保留)"
echo "========================================"
sleep 3
shutdown now
