import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 步骤1：准备可调节过渡程度的示例数据（第一象限内） ----------------------
# 混合权重参数：blend_weight（取值范围[0, 1]）
# 0 = 完全随机分布（与原代码一致）
# 1 = 完全特定规律分布（此处选择线性正相关规律，可按需修改）
# 0~1之间 = 随机与特定规律的混合过渡状态
blend_weight = 0.4  # 可修改此值体验过渡效果

# 生成基础横坐标数据（100个样本，范围0-10）
x = np.random.uniform(1, 10, 100)

# 1. 生成完全随机的纵坐标数据（与原代码一致）
y_random = np.random.uniform(1, 10, 100)

# 2. 生成完全特定规律的纵坐标数据（此处选择线性规律：y = 0.8x + 1，加少量噪声更自然）
# 若需其他特定分布（如二次函数、正态分布等），可修改此部分
y_trend = 0.8 * x + 1 + np.random.normal(0, 0.5, 100)  # 线性趋势+小幅噪声
# 限制特定规律数据在第一象限合理范围内（避免超出0-10）
y_trend = np.clip(y_trend, 0.5, 10)

# 3. 按混合权重融合随机数据与特定规律数据，实现平滑过渡
# 核心公式：混合数据 = 随机数据*(1-权重) + 特定规律数据*权重
y = (1 - blend_weight) * y_random + blend_weight * y_trend

# ---------------------- 步骤2：创建画布和坐标系 ----------------------
fig, ax = plt.subplots(figsize=(8, 6))  # figsize设置图表大小（宽, 高）

# ---------------------- 步骤3：绘制散点图（70个黑色，30个红色） ----------------------
# 方法1：拆分数据绘制（推荐，逻辑清晰，与原代码风格一致）
# 前70个点保持黑色
ax.scatter(x[:70], y[:70], s=80, c='black', alpha=0.9)
# 后30个点设置为红色
ax.scatter(x[70:], y[70:], s=80, c='red', alpha=0.9)

# 方法2：颜色数组法（可选，适合扩展多颜色场景）
# colors = ['black'] * 70 + ['red'] * 30
# ax.scatter(x, y, s=80, c=colors, alpha=0.9)

# 方法3：随机选择30个点为红色（可选，避免固定前后切片的规律性）
# red_indices = np.random.choice(100, 30, replace=False)
# black_indices = np.setdiff1d(np.arange(100), red_indices)
# ax.scatter(x[black_indices], y[black_indices], s=80, c='black', alpha=0.9)
# ax.scatter(x[red_indices], y[red_indices], s=80, c='red', alpha=0.9)

# ---------------------- 步骤4：调整刻度设置（移除所有刻度，包括纵坐标0刻度） ----------------------
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])

# ---------------------- 步骤5：仅保留第一象限的横纵坐标轴（无四周边框、无负向线） ----------------------
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

# ---------------------- 步骤6：给第一象限横纵坐标轴添加箭头（指向第一象限外侧） ----------------------
x_max = ax.get_xlim()[1]  # 获取x轴当前最大值
ax.plot(x_max, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
y_max = ax.get_ylim()[1]  # 获取y轴当前最大值
ax.plot(0, y_max, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# ---------------------- 步骤7：确保不显示图例 ----------------------
ax.legend_.remove() if ax.legend_ else None

# ---------------------- 步骤8：调整坐标范围（仅保留第一象限，无负向区间） ----------------------
ax.set_xlim(0, 11)
ax.set_ylim(0, 11)

# ---------------------- 步骤9：添加混合权重标注（可选，方便查看当前过渡状态） ----------------------
# ax.text(0.5, 10.5, f"blend_weight = {blend_weight}", fontsize=12, ha='center')

# ---------------------- 步骤10：显示图表 ----------------------
plt.show()