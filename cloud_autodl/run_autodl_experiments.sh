#!/bin/bash
# ============================================================
# DACO 超参数实验批量运行脚本 (AutoDL)
# ============================================================
# 论文 Section IV-D "Hyperparameter Analysis" 需要的实验:
#   1. RCAJS β_p: 7 个取值 × VGG-7     (Fig. hyper_rcajs(a))
#   2. RCAJS φ_b: 5 个取值 × VGG-7     (Fig. hyper_rcajs(b))
#   3. MCSS κ:    7 个取值 × VGG-7     (Fig. hyper_mcss)
# 共 19 次完整训练 (200 epochs + 10 warmup + 20 cooldown)
#
# 预估 GPU 时间 (RTX 3090):
#   VGG-7 CIFAR-10 200ep ≈ 30~50 min/trial
#   19 trials × 40 min ≈ 13 GPU-hours
#
# 使用方式:
#   nohup bash autodl/run_autodl_experiments.sh \
#     > outputs/autodl_experiments/logs/batch.log 2>&1 &
#   tail -f outputs/autodl_experiments/logs/batch.log
# ============================================================
set -e

PROJECT_DIR="/root/autodl-tmp/geta"
cd "$PROJECT_DIR"

# 激活 conda 环境
source activate daco 2>/dev/null || conda activate daco

OUT_DIR="outputs/autodl_experiments"
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/results" "$OUT_DIR/checkpoints"

EPOCHS=${EPOCHS:-200}
MODEL=${MODEL:-vgg7}
DATA_DIR=${DATA_DIR:-/root/autodl-tmp/data}

echo "========================================"
echo "  DACO 超参数实验批量运行"
echo "  Model: $MODEL, Epochs: $EPOCHS"
echo "  Start: $(date)"
echo "========================================"

# ---- 0. 冒烟测试 (验证流程) ----
echo ""
echo "[0/4] 冒烟测试 (1 epoch)..."
python run_daco_cifar10.py \
    --model "$MODEL" --epochs 1 --quick-test \
    --data-dir "$DATA_DIR" --out "$OUT_DIR"
echo "[0/4] 冒烟测试完成: $(date)"

# ---- 1. β_p 敏感性扫描 (7 个取值) ----
echo ""
echo "[1/4] RCAJS β_p 敏感性扫描..."
echo "  开始: $(date)"
python run_daco_cifar10.py \
    --model "$MODEL" --hparam beta_p --epochs "$EPOCHS" \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt
echo "  完成: $(date)"

# ---- 2. φ_b 敏感性扫描 (5 个取值) ----
echo ""
echo "[2/4] RCAJS φ_b 敏感性扫描..."
echo "  开始: $(date)"
python run_daco_cifar10.py \
    --model "$MODEL" --hparam phi_b --epochs "$EPOCHS" \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt
echo "  完成: $(date)"

# ---- 3. κ 敏感性扫描 (7 个取值) ----
echo ""
echo "[3/4] MCSS κ 敏感性扫描..."
echo "  开始: $(date)"
python run_daco_cifar10.py \
    --model "$MODEL" --hparam kappa --epochs "$EPOCHS" \
    --data-dir "$DATA_DIR" --out "$OUT_DIR" --save-ckpt
echo "  完成: $(date)"

# ---- 4. 收集结果 ----
echo ""
echo "[4/4] 收集结果..."
python collect_results.py --out "$OUT_DIR"

echo ""
echo "========================================"
echo "  全部实验完成!"
echo "  End: $(date)"
echo "  结果目录: $OUT_DIR"
echo "========================================"
echo ""
echo "下一步:"
echo "  1. 查看汇总 CSV: cat $OUT_DIR/experiment_summary.csv"
echo "  2. 下载结果到本地填入 make_hyper_figs.py"
echo "  3. 或直接 scp: scp -r root@<autodl-ip>:$PROJECT_DIR/$OUT_DIR/results/ ./"
