# GETA/DACO 工作日志

> 记录每次开发的改动、结论与踩坑，便于回溯。
> 分支：`feature/daco` ｜ 环境：conda `d2l`（torch 2.1.0+cu118, RTX 4060 Laptop）

---

## 2026-11-02 — 定位并修复「量化层失效」，跨框架对比实验

### 一、问题现象

100 epoch 实验（15 warmup + 85 joint）从第 1 到第 37 epoch：

- `val_acc` 恒为 **0.1000**（10 类随机水平）
- `loss` 恒为 **2.3**（= ln10，随机猜测）
- warmup 阶段（纯全精度逻辑）就完全学不动

### 二、根因（逐层实测定位，非推测）

**根因 1（主因）：实验脚本用错了 optimizer variant**

GETA 的量化参数 `d_quant / q_m / t_quant` 梯度是**对整个 tensor 求和**，
量级天然可达 **1e9**。实验脚本用 `variant="sgd"` + `lr=0.1`，一步就把：

| 参数 | 修复前 | 一步后 | 后果 |
|---|---|---|---|
| `d_quant_wt` (conv1) | 2.5e-5 | 1.9e6 | 量化步长爆炸 |
| `t_quant_wt` (conv1) | 1.0 | -171021 | 非线性映射失控 |
| `conv1.weight` | 0.82 | 1.46e8 | 权重本身被污染 |
| 量化后权重 | absmean 0.216 | **100% 归零** | 整层输出塌陷 |

**官方 README 示例用的是 `variant="adam"`**，Adam 的 `grad/sqrt(v)` 自适应缩放
正好能吸收这个量级。改回 adam 后，1 epoch `val_acc=0.2396`，5 epoch 内持续上升。

**根因 2：本分支把「崩溃」包装成了「静默失效」**

- 原版 `quant_layers.backward` 末尾检测 grad_t 的 NaN 并
  `raise NanInGradientError` → 立刻暴露
- 本分支改成 `torch.nan_to_num(grad, 0.0, 0.0, 0.0)` 全清零后继续训练
  → 权重被一步污染到 1e8，表象只剩「acc 恒随机」，根因完全不可见

**根因 3：forward 溢出被置零**

`output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)`
把溢出输出置 0（应饱和到 ±|q_m|），会让整个量化层输出塌陷为 0。

**根因 4：`importance_score` 出现第二次归一化**

本分支在 `calculate_importance_score` 内部新增 `_apply_online_normalization()`：
用**按 criterion 名索引的全局 min-max** 归一化。因为 bounds 的 key 是
criterion 名而非层名，所有层共用同一组极值：

- 幅值小的层（如 layer1 的 1e-5 量级）被压到 ~0
- 剪枝时按分数排序 → 整层被剪空

原版只在 `BaseHybridSparseOptimizer.compute_importance_scores` 里做一次
Frobenius 范数归一化（`sqrt(sum(score²))`），有理论依据且保持层间可比性。

**根因 5：`_GLOBAL_SCORE_BOUNDS` 清理 key 用错**

```python
if layer_name in _GLOBAL_SCORE_BOUNDS:   # 字典 key 是 cri_name
    del _GLOBAL_SCORE_BOUNDS[layer_name]  # 永远不成立
```
→ 跨实验污染，第二个实验继承上一个实验的极值。

### 三、修复内容（commit `c161058`）

**`only_train_once/quantization/quant_layers.py`**

- 溢出输出饱和到 ±`|q_m|`，不再置 0
- NaN 处理改为三档策略，环境变量 `OTO_QUANT_NAN_POLICY` 控制：
  - `warn`（默认）：采样告警（每 500 次）+ 清零，日志带 `[Quant][NaN]`
  - `strict`：恢复原版 `raise NanInGradientError`
  - `silent`：复现历史静默行为（仅用于对照复现）
- 采样检查的原因：`torch.isfinite(t).all()` 每层每步做全张量归约 +
  **隐式 GPU→CPU 同步**，会让训练再慢 18%

**`only_train_once/optimizer/importance_score/__init__.py`**

- 内部 min-max 归一化默认关闭（`OTO_ONLINE_NORM=1` 可开启）
- 新增 `reset_score_state()`，供每个实验开始时清空跨实验状态
- 修正 bounds 清理逻辑（整体清空 + 每个 step 只清一次）

**`run_daco_vs_baseline.py`**

- `variant` 默认 `adam` / `lr` 默认 `1e-3`
- 新增 `--root`：同一份实验代码可加载任意框架副本，用于跨框架公平对比
- DACO 特性注入（RCAJS/MCSS/DGD）改为 `hasattr` 探测，兼容无这些属性的原版
- 新增 tensor 版 CIFAR-10 数据管道 `_FastCIFARLocal`
  （原先 PIL 逐张解码 ~60s/epoch，现在 0.5ms/step）

### 四、性能优化（commit `d193756`）

量化训练比纯浮点慢 7.4 倍，逐段 profiling 后逐项优化：

| 阶段 | 纯浮点 | 优化前 | 优化后 | 手段 |
|---|---|---|---|---|
| forward | 6.9 ms | 44.6 ms | 31.5 ms | `pow` 替代 `exp(t*log(x))`、`where` 替代布尔索引 |
| backward | 15.4 ms | 114.0 ms | 73.1 ms | 复用 mask / `grad_output*sign`、标量先归约再乘、`masked_fill` |
| opt.step | 3.3 ms | 35.3 ms | 34.8 ms | 基本无优化空间（GETA 白盒逐参操作） |
| **合计** | **26 ms** | **194.4 ms** | **141.6 ms** | **-27%** |

参考：**微软原版同配置 165.1 ms/step，现在我们比原版快 14%**。
（76 → 55 s/epoch）

**为什么停在 141.6 ms**：瓶颈已转为**显存带宽**。每层 backward 有约 30 个
全张量 kernel，2M 元素的读写使「减少 kernel 数量」的边际收益趋近于 0
（实测提取公因式只从 141.9 → 141.6 ms）。再提速需算子融合
（`torch.compile` 或自定义 CUDA kernel），不在本次范围。

### 五、等价性验证方法（重要，可复用）

`verify_consistency.py`：固定 seed + `shuffle=False` 固定数据顺序，逐 step 记录 loss。

**判据不是「两次完全相等」**，而是对比两组差异是否同量级：

| 对比 | 首步 loss | 60 步最大漂移 |
|---|---|---|
| 优化前后 | 2.447822（完全相同） | 2.0% |
| **同一份代码跑两次**（基准） | 2.447822 | **1.0%** |

两者同量级 → 漂移来自 GPU 非确定性（cuDNN 反向 `atomicAdd` 求和顺序），
不是实现差异。另有公式级验证：forward `max diff = 0.0`，
backward 在 30 组跨数量级随机配置下 `max diff = 2.3e-5`
（梯度量级 ~1e9，相对 1e-14）。

### 六、跨框架对比实验

**分支认知纠正**：本地 `main`（5963445）其实是 daco 线的 initial commit，
**不是微软原版**。真正的原版是 `origin/main` = `0518a57 remove redundant code`，
两者**无共同祖先**。已把本地 `main` 对齐到 `origin/main`。

原版基座用 worktree 导出：
```bash
git worktree add ../geta_origin_main origin/main
```

**实验设置**（两边完全一致，只有框架实现不同）：

- 模型 ResNet-20（`sanity_check/backends/resnet20_cifar10.py`，两份仅差
  `feature_need` 可选参数，结构与初始化一致）
- 量化 `WEIGHT_AND_ACTIVATION`，adam / lr 1e-3 / lr_quant 1e-3
- 15 epoch warmup + 85 epoch joint，每 epoch 剪枝一次
- `target_group_sparsity=0.5`，CIFAR-10 / bs 128

```bash
# 我们 (DACO: RCAJS beta_p=1.0, MCSS kappa=0.5, DGD noise=1e-4)
python run_daco_vs_baseline.py --warmup-epochs 15 --epochs 85 \
    --modes daco --out outputs/framework_compare/ours

# 微软原版 (纯 GETA)
python run_daco_vs_baseline.py --warmup-epochs 15 --epochs 85 \
    --modes baseline --out outputs/framework_compare/origin \
    --root E:/pythonProject/geta-main/geta_origin_main
```

**绘图**：
```bash
python plot_framework_compare.py --origin-dir outputs/framework_compare/origin \
    --ours-dir outputs/framework_compare/ours --out outputs/framework_compare
```
产出 `fig_curves.png`（准确率 / 稀疏度 / 压缩率 三子图）、
`fig_tsne_trajectories.png`（逐步类别原型迁移轨迹）、
`fig_drift_metric.png`（每 epoch 平均位移，量化「平滑 vs 震荡」）。

### 七、原型增量落盘（20:36 补）

**问题暴露**：第一轮正式实验跑到约 20 分钟被中断，`outputs/framework_compare/`
下**一个文件都没有** —— 因为原逻辑把 `prototypes_seq` 全部攒在内存里，
只在训练**全部结束后**才 `np.save`。中断即全丢，改图必须重跑。

**修复**：新增 `_flush_epoch_artifacts()`，在**每个 epoch 结束**立即覆盖写：

- `prototypes/{model}_{mode}_protos.npy` — `(T_sofar, 10, D)` 原型序列
- `{mode}_history.jsonl` — 每行一个 epoch 的全部指标

覆盖写而非追加：与内存里的 history 严格同步（幂等），文件很小
（100 epoch 的 npy 约 256 KB，jsonl 约 20 KB），写盘开销可忽略。
落盘异常只告警、不打断训练。

这样任何时刻磁盘上都有「到目前为止」可直接画图的数据，
换图形式（折线 → 堆叠 / 散点 → 轨迹 / 加子图）都不需要重跑训练。

### 八、踩坑记录

1. **数据读取**：`pickle.load(encoding="bytes")` 会得到 `b"data"`，本项目的
   CIFAR-10 是 `encoding="latin1"` + str key（`"data"` / `"labels"`）
2. **文件被临时覆盖**：做一致性对照时把原始版 `quant_layers.py` 复制进工作区，
   命令被中断导致优化版丢失。教训：对比测试不要在活工作区替换文件，
   改用 patch `sys.modules` 或独立目录。备份 `/tmp/ql_optimized.py` 救了回来。
3. **`sleep` 超过工具默认 timeout 会被 SIGTERM**：默认 120s，长等待要拆成
   多次短 sleep 或设 `timeout`
4. **`| grep` 会块缓冲**：后台任务的 `print(flush=True)` 被 grep 吃掉，
   中途看不到进度。要实时监控需 `grep --line-buffered`
5. **磁盘**：项目在 E 盘（可用仅 20G），D 盘 218G。逐步存 ckpt（1.2MB × 39000
   ≈ 47GB）会撑爆 E 盘；每 epoch 存（~120MB）才安全
6. **只在实验结束落盘 = 中断即前功尽弃**：第一轮正式实验跑 20 分钟被中断后
   目录全空。长跑实验必须逐 epoch 增量落盘（见第七节），否则一次中断就要
   重跑几小时
7. **后台任务会被会话中断连带杀掉**：`run_in_background` 的任务在用户打断
   当前轮次时会被 cancel，长跑实验要注意这一点

### 九、待办

- [ ] 重启正式实验（已具备逐 epoch 增量落盘，中断不再丢数据）
- [ ] 等两个实验跑完，出图并分析 DACO vs 原版的特征漂移平滑度
- [ ] 视结果决定是否需要 ckpt 逐步保存（当前只在结束时存 `final.pt`）
- [ ] 工作区 47 个未跟踪文件（实验产物）待整理，建议加 `.gitignore`：
      `outputs/ temp/ datasets/ cache/ *_result*.txt __pycache__/`
