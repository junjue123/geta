# DACO 超参数实验脚本使用说明

`run_hparam_sweep.py` 是一个轻量级、可复现的超参数扫描工具，用于验证 **DACO**
三大核心模块（**RCAJS** 资源约束自适应联合稀疏 / **MCSS** 多标准校准重要性分数 /
**DGD** 扩散梯度下降 / 朗之万噪声）在 VGG7 + GETA（权重与激活联合量化）上的行为差异。

> 设计目标：不依赖 GPU、不做完整 ONNX trace，仅在 CPU 上跑 12 步短训练，
> 记录逐 step 的 Loss / 组稀疏度 / 平均权重位宽轨迹，并自动出对比图，
> 让超参效果“看得见”。

---

## 1. 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | 3.8+ | 推荐 `d2l` conda 环境（torch 2.1.0） |
| torch | 2.0+ | CPU 即可运行 |
| matplotlib | 3.x | 仅绘图时需要（可 `--no-plot` 跳过） |

本仓库自带 GETA/OTO 框架，无需额外安装；脚本会把项目根目录加入 `sys.path`。

**推荐运行命令（指定 conda 环境）：**

```bash
cd geta-main_origin
CUDA_VISIBLE_DEVICES="" /d/softwares/Anaconda/envs/d2l/python.exe run_hparam_sweep.py --module all --quick
```

> 若使用默认 `python`，请确认其环境中已安装 `torch` 与 `matplotlib`。

---

## 2. 快速开始

```bash
# (1) 最快冒烟测试：每个模块只跑前 2 个配置，约 3 分钟，产出 CSV+MD+3 张图
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all --quick

# (2) 完整扫描：RCAJS + MCSS + DGD 共 25 组配置，约 15 分钟
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all

# (3) 只扫单个模块
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module rcajs
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module mcss
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module dgd

# (4) 自定义训练步数（默认 12 步，短训练）
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module rcajs --steps 20

# (5) 无 matplotlib 环境：仅产出 CSV + Markdown，不出图
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all --no-plot

# (6) 指定输出目录
CUDA_VISIBLE_DEVICES="" python run_hparam_sweep.py --module all --out outputs/hparam_sweep
```

---

## 3. 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--module` | `all` | 扫描模块：`rcajs` / `mcss` / `dgd` / `all` |
| `--steps` | `12` | 每个 trial 的训练步数（短训练，不追求收敛） |
| `--device` | `cpu` | 运行设备（脚本强制 `CUDA_VISIBLE_DEVICES=""`，仅 CPU） |
| `--out` | `outputs/hparam_sweep` | 结果输出目录（CSV / MD / PNG） |
| `--quick` | 关 | 快速模式：每个模块只跑前 2 个配置 |
| `--no-plot` | 关 | 不生成对比图 |

---

## 4. 扫描内容（各模块可控超参）

### 4.1 RCAJS（资源约束自适应联合稀疏）
- `target_group_sparsity`：目标组稀疏度（0.3 / 0.5 / 0.7）
- `pruning_periods`：剪枝周期数（1 / 3 / 5）→ 控制稀疏度爬升节奏
- `bit_reduction`：每周期位宽缩减步长（1 / 2 / 4）
- `max_bit_wt`：权重最大位宽（8 / 16 / 32）
- `min_bit_wt`：权重最小位宽（2 / 4）

> 注：RCAJS 的 **β_p（阻尼系数）** 与 **φ_b（收缩比）** 论文超参分析见
> 专用脚本 `run_rcajs_hparam.py`（GETA 优化器内部已接线 `RCAJSController`，
> β_p 经 `opt.rcajs.beta_p` 生效、φ_b 由训练循环外部调度器驱动 `opt.max_bit_wt`）。

### 4.2 MCSS（多标准校准重要性分数）
- `mcss_smooth_factor`：校准因子混合比例（0.5 / 0.8 / 1.0）
- `mcss_history_window`：稳定性因子历史窗口（3 / 5 / 10）

> 框架原生的 `compute_importance_scores` 未把 `smooth_factor` / `history_window`
> 暴露为优化器参数。本脚本通过运行时 patch `calculate_importance_score`
> 注入 MCSS 校准逻辑，保证扫描真实生效（research 模式）。

### 4.3 DGD（扩散梯度下降 / 朗之万噪声）
- `diffusion_noise_init`：初始朗之万噪声幅度（1e-5 / 1e-4 / 1e-3）
- `diffusion_noise_ratio`：噪声衰减比（0.1 / 0.5 / 0.9）
- `diffusion_steps`：扩散步数（默认 = `pruning_steps`）

> `oto.geta()` 是固定签名 wrapper，不转发 DGD 超参；脚本在构造优化器后
> 将这些参数作为属性注入（`opt.diffusion_noise_init` 等），由 GETA 的
> `step()` 读取。

---

## 4.4 论文「Hyperparameter Analysis」专项：`run_rcajs_hparam.py`

论文 Section IV「Hyperparameter Analysis」显式要求分析 **RCAJS 的 β_p 与 φ_b**
两个超参的敏感性（对应论文 Fig. hyper_rcajs(a)/(b)）。本仓库的
`run_rcajs_hparam.py` 专门补齐这两项此前欠缺的实验：

| 论文超参 | 含义 | 取值扫描 | 生效方式 |
|---|---|---|---|
| **β_p** | 阻尼系数，经 `r_p = r_base·exp(-β_p·Δ_p)` 调节自适应剪枝率 | `0.01 / 0.05 / 0.1 / 0.3 / 0.5 / 1.0 / 2.0` | 构造后注入 `opt.rcajs.beta_p` |
| **φ_b** | 收缩比阈值，低于 max_bit 的层比例超过阈值则触发一次位宽收缩（受 `min_bit_wt` 硬下界保护） | `0.1 / 0.3 / 0.5 / 0.7 / 0.9` | 训练循环外部门控调度器驱动 `opt.max_bit_wt` |

### 启用方式

```bash
# (1) 跑全部 RCAJS 论文超参 (β_p + φ_b)
CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py

# (2) 只扫 β_p (论文 Fig. hyper_rcajs(a))
CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode beta

# (3) 只扫 φ_b (论文 Fig. hyper_rcajs(b))
CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode phi

# (4) 快速冒烟 (β_p/φ_b 各取 3 个代表值)
CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --quick

# (5) 自定义步数 / 输出目录 / 跳过绘图
CUDA_VISIBLE_DEVICES="" python run_rcajs_hparam.py --mode all --steps 20 \
    --out outputs/rcajs_hparam --no-plot
```

### 说明
- 单变量隔离：β_p / φ_b 均固定 `target_group_sparsity=0.5`，避免互相干扰（论文主实验默认 φ_b=0.9、b_ε=4、κ=0.1、W=5）。
- 代理精度：随机数据下 `proxy_acc = 1/(1+final_loss)`，仅用于观察相对趋势；论文真实曲线需以 CIFAR-10 全量训练替换 `run_trial` 中的训练循环。
- 输出：`rcajs_hparam_<mode>_<时间戳>.csv` / `.md`，以及 `plot_rcajs_beta.png`、`plot_rcajs_phi.png`、`plot_rcajs_phi_traj.png`。

---

## 5. 输出说明

结果写入 `--out` 目录（默认 `outputs/hparam_sweep/`，已被 `.gitignore` 忽略）：

| 文件 | 内容 |
|---|---|
| `sweep_<module>_<时间戳>.csv` | 全部 trial 的指标表（组稀疏度、零组数、平均/最小位宽、Loss、耗时、状态） |
| `sweep_<module>_<时间戳>.md` | Markdown 汇总报告，含每模块结果表与解读 |
| `plot_rcajs.png` | RCAJS 模块对比图：Loss / 组稀疏度 / 平均权重位宽 三子图轨迹 |
| `plot_mcss.png` | MCSS 模块对比图 |
| `plot_dgd.png` | DGD 模块对比图 |

**轨迹图读法：**
- **Loss 子图**：应随 step 下降，说明模型在学。
- **组稀疏度子图**：剪枝 `commit` 后（通常 step 7 附近）会阶跃上升；不同 `pruning_periods` 决定最终稳定稀疏度。
- **平均权重位图**：量化生效后（通常 step 8 附近）骤降；`max_bit_wt` / `min_bit_wt` 决定上下界。

---

## 6. 稳健性设计（避免误报 FAIL）

- **确定性种子**：每个 trial 用 `seed_base + attempt` 初始化，结果可复现。
- **失败重试**：GETA 在极少数随机初始化下会触发潜在除零（数据相关），
  脚本自动换种子重建重跑，最多 3 次（`max_retry`）。
- **跨 trial 全局态隔离**：每个 trial 前调用 `reset_mcss_history()` 清空
  `importance_score` 模块的 `_SCORE_HISTORY` 与 `_GLOBAL_SCORE_BOUNDS`，
  避免稳定性因子 / 归一化边界跨 trial 累积污染。
- **框架层防御**：已在 `geta.py` / `geta_b.py` 的 `_d_quant_helper` 中对
  `bit_width == 1` 的除零（`2**(bw-1)-1 == 0`）做钳制（`eff_bw = max(bw, 2)`）。

---

## 7. 已知限制

1. 短训练（默认 12 步）仅用于**功能验证与超参趋势观察**，不代表最终压缩率/精度。
2. 默认随机输入（`torch.randn`）且 batch 极小，Loss 绝对值无实际意义，关注**相对趋势**。
3. MCSS 接线为运行时 patch，若框架后续原生支持 `smooth_factor` / `history_window`，
   应改为直接传参，移除 patch 逻辑。
4. `outputs/` 目录已被 `.gitignore` 忽略，扫描结果不会进入版本库。

---

## 8. 常见排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: only_train_once` | 未在项目根目录运行 | `cd` 到 `geta-main_origin/` 再运行 |
| 长时间无输出 | 首次构建 OTO 计算图较慢 | 耐心等待，或加 `--quick` 先冒烟 |
| 全部 `FAIL` 且含 `ZeroDivisionError` | 框架未应用除零修复 | 确认 `geta.py` / `geta_b.py` 含 `eff_bw` 钳制 |
| 绘图报 `findfont` 中文方块 | matplotlib 缺中文字体 | 不影响数值，或安装 `SimHei` 字体 |

---

## 9. 相关文件

- 主脚本：`run_hparam_sweep.py`
- 论文超参专项：`run_rcajs_hparam.py`（RCAJS β_p / φ_b 敏感性分析）
- 框架修复：`only_train_once/optimizer/geta.py`、`geta_b.py`、`graph/utils.py`、`graph/graph.py`、`optimizer/importance_score/__init__.py`
- 模型：`sanity_check/backends/vgg7.py`
- 验证用例：`only_train_once/tests/verify_daco_fixes.py`
