# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GETA (Generic, Efficient Training framework that Automates joint structured pruning and mixed precision quantization) is a PyTorch-based neural network compression framework. It automates structured pruning and quantization across CNNs, transformers, vision transformers, and small language models.

## Core Architecture

The framework follows a graph-based approach to neural network compression:

- **OTO (Only Train Once)**: Main entry point class that orchestrates the entire compression pipeline
- **Graph Analysis**: Constructs pruning search space via quantization-aware dependency graph (QADG)
- **GETA Optimizer**: White-box optimizer for joint sparsity and quantization constraints (QASSO)
- **Subnet Construction**: Extracts compressed subnetworks after training

Key modules in `only_train_once/`:
- `graph/`: Computational graph construction and node grouping
- `optimizer/`: GETA, HESSO, and other sparse optimizers
- `quantization/`: Quantization-aware layers and model conversion
- `transform/`: Graph and tensor transformations
- `subnet_construction/`: Pruned model extraction

## Development Commands

### Running Tests
```bash
# Run all sanity checks
cd sanity_check
python sanity_check.py

# Run individual test
python -m pytest sanity_check/test_resnet18.py -v

# Run specific test class
python -m pytest sanity_check/test_resnet18.py::TestResNet18::test_sanity -v
```

### Installation
```bash
git clone https://github.com/microsoft/geta.git
cd geta
pip install -e .
```

Requires PyTorch >= 2.0

## Usage Pattern

```python
from only_train_once import OTO
from only_train_once.quantization.quant_model import model_to_quantize_model

# 1. Prepare quantized model
model = model_to_quantize_model(model, quant_mode="weight_and_activation")

# 2. Initialize OTO with dummy input
dummy_input = torch.rand(1, 3, 32, 32)
oto = OTO(model=model.cuda(), dummy_input=dummy_input.cuda())

# 3. Create optimizer with compression targets
optimizer = oto.geta(
    variant="adam",
    lr=1e-3,
    target_group_sparsity=0.5,
    bit_reduction=2,
    min_bit_wt=4,
    max_bit_wt=16,
    # ... other parameters
)

# 4. Train normally
for epoch in range(max_epoch):
    for X, y in trainloader:
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. Extract compressed subnet
oto.construct_subnet(out_dir='./')
```

## Key Concepts

- **Node Groups**: Parameters grouped for joint pruning decisions
- **Group Sparsity**: Fraction of node groups zeroed out
- **Bit Width Range**: Target quantization precision bounds
- **Projection/Pruning Steps**: Training steps for different compression phases

## File Structure

- `sanity_check/`: Test suite with model-specific test cases
- `tutorials/`: Jupyter notebooks with usage examples
- `test_scripts/`: Extended test scripts for various architectures
- `tools/`: Visualization utilities

## Training Entry & Configs

**唯一正式入口是 `train.py`，所有配置都在 YAML 里改。**

```bash
python train.py configs/compare_resnet20.yaml      # 正式跑
python train.py configs/compare_vgg7.yaml          # 换模型只改 yaml 的 model
python train.py configs/xxx.yaml --modes daco      # 换算法只改 modes
python train.py configs/xxx.yaml --quick-test      # 冒烟: 1 epoch, 无 warmup/cooldown
```

优先级：**命令行 > YAML > 脚本内置默认值**。运行开始会把最终生效的完整配置快照到
`{out}/run_config.yaml`，便于事后回溯。

### 常用改动都在 YAML 内

| 想改什么 | 改哪个字段 | 取值 |
| --- | --- | --- |
| 换模型 | `model` | `resnet20` / `vgg7` |
| 换算法 | `modes` | `baseline`（纯 GETA）/ `daco`（余弦剪枝）/ `baseline,daco`（对比） |
| 量化模式 | `quant_mode` | `weight_and_activation` / `weight_only` |
| 三段调度 | `warmup_epochs` / `epochs` / `cooldown_epochs` | warmup 纯 SGD → joint 剪枝降位宽 → cooldown 冻结结构 finetune |
| 优化器 | `variant` / `lr` / `lr_quant` / `weight_decay` | 必须用 `adam`：量化参数量级 ~1e9，sgd 会一步爆炸 |
| 剪枝形状 | `prune_shape` / `prune_floor` / `prune_budget` | DACO 用 `cosine`，baseline 恒为 `linear` |
| 压缩目标 | `compression.*` | `target_group_sparsity` / `bit_reduction` / 位宽上下界 / `kappa` / `beta_p` 等 |
| 数据集 | `dataset.*` | 路径、batch size、归一化、增强 |
| 输出 | `out` / `ckpt_every` | 结果目录与存档频率 |

新增一个实验：复制 `configs/compare_resnet20.yaml`，改 `model` / `modes` / 超参即可，
不需要写新脚本。

### 目录结构

```
geta-main_origin/
├── train.py              # 唯一训练入口
├── configs/              # YAML 配置 (正式) + legacy/ (旧格式)
├── only_train_once/      # 核心框架包 (optimizer / quantization / graph / distillation ...)
├── sanity_check/         # 测试后端与模型定义
├── scripts/              # 工具脚本: 诊断 diag_*, 测试 run_*_test, 性能 profile_*,
│                         #          超参扫描, 数据下载, CR 统计等
├── tools/                # 可视化与出图: plot_framework_compare, plot_baseline_vs_daco,
│                         #              collect_results, make_picture*
├── deployment/           # cuda_kernels/ (生成的 CUDA 核) + frontend/ (配置 UI)
├── cloud_autodl/         # 云端训练脚本
├── docs/                 # 实验报告与说明 (本地 *.md 规则下不入库)
├── legacy/               # 已弃用脚本
├── outputs/              # 训练产物 (gitignore)
└── temp/                 # 运行期打分日志与中间图 (gitignore)
```

`scripts/` 下的脚本用 `os.path.dirname(os.path.dirname(__file__))` 定位项目根，
可以任意层级调用（如 `python scripts/compute_cr.py`）。

### 已弃用的入口

- `legacy/run_daco_cifar10.py`：旧训练入口，其 `cooldown_epochs` 已并入 `train.py`；
  仅保留用于 `--hparam/--hvalue` 超参扫描的历史复现。旧配置在 `configs/legacy/` 下。
- `configs/legacy/*.yaml`：给旧入口用的格式，不要用于新实验。

## Commit Message Convention

每次提交必须遵循 `[<type>] <subject>` 格式：方括号内为改动类型，空一格后写描述。

```text
[feat] GETA 优化器加入 Langevin 噪声注入
[fix] 修正断点续训时学习率退火被重置的问题
[chore] 移除 WORKLOG.md 的版本跟踪
```

规则：

1. **必须以方括号类型标签开头**，类型小写，其后恰好一个空格再接描述。
2. **描述用祈使句**，说明"做了什么"，不加句号，建议 50 字以内。
3. **一次提交只做一件事**，不要把功能、修复、格式化混在同一个 commit 里。
4. 需要限定范围时可加可选的 scope：`[feat](plot) 双联合并图 fig_combined`，不强制。
5. 禁止无意义的提交信息（如 `update`、`修改`、`fix bug`、`.`）。

类型清单：

| 类型 | 用途 |
| --- | --- |
| `feat` | 新增功能 |
| `fix` | 修复缺陷 |
| `docs` | 仅文档变更 |
| `style` | 格式调整（空白、缩进、分号），不影响逻辑 |
| `refactor` | 重构，不改变外部行为 |
| `perf` | 性能优化 |
| `test` | 测试用例或测试脚本 |
| `chore` | 构建、依赖、配置、杂项清理 |
| `revert` | 回滚此前的提交 |

示例（对应本仓库的实际工作）：

```text
[feat] DACO 双联对比图输出 fig_combined.png
[fix] 重要性打分跨实验状态未清空导致分数漂移
[perf] 逐层 BOPs 统计改为缓存累计
[chore] 停止跟踪训练产物 scores_log_raw.txt
```
