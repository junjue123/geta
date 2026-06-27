import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 步骤1：基础参数与原始数据生成 ----------------------
blend_weight = 0.6
n_target_points = 30
target_value = 50
approach_strength = 1

# 新增：强避开效果的控制参数（关键优化）
safe_margin = 5  # 危险区间：50±10（40~60），安全区间：<40 或 >60
avoid_force = 1  # 强制推开的距离，值越大，普通点离50越远

# 生成基础数据
x = np.random.uniform(1, 100, 100)
y_random = np.random.uniform(1, 100, 100)
y_trend = 0.8 * x + 1 + np.random.normal(0, 0.5, 100)
y = (1 - blend_weight) * y_random + blend_weight * y_trend

# ---------------------- 目标点处理（保持原有逻辑） ----------------------
target_indices = np.random.choice(len(y), size=n_target_points, replace=False)
y[target_indices] = (1 - approach_strength) * y[target_indices] + approach_strength * target_value

# ---------------------- 核心：强效果让普通点避开50（彻底解决效果不明显问题） ----------------------
mask = np.ones(len(y), dtype=bool)
mask[target_indices] = False  # 筛选普通点
y_normal = y[mask]  # 提取普通点

# 强排斥逻辑：遍历每个普通点，强制推离危险区间（40~60）
for i in range(len(y_normal)):
    current_val = y_normal[i]
    # 1. 若点在 50-safe_margin ~ 50 之间（40~50），强制推到 50-safe_margin - avoid_force（<30）
    if (target_value - safe_margin) <= current_val <= target_value:
        y_normal[i] = (target_value - safe_margin) - avoid_force
    # 2. 若点在 50 ~ 50+safe_margin 之间（50~60），强制推到 50+safe_margin + avoid_force（>70）
    elif target_value < current_val <= (target_value + safe_margin):
        y_normal[i] = (target_value + safe_margin) + avoid_force
    # 3. 已经在安全区间的点，保持不变（无需额外处理）

# 将修正后的普通点赋值回原数组
y[mask] = y_normal

# ---------------------- 数据分离与绘图（保持原有逻辑） ----------------------
x_target = x[target_indices]
y_target = y[target_indices]
x_normal = x[mask]
y_normal = y[mask]

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制散点（先普通点，后目标点）
ax.scatter(x_normal, y_normal, s=80, c='black', alpha=0.9)
ax.scatter(x_target, y_target, s=80, c='red', alpha=0.9)

# 坐标轴美化（保持原有逻辑）
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_yticks([])
ax.set_yticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')

# 添加坐标轴箭头
x_max = ax.get_xlim()[1]
ax.plot(x_max, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
y_max = ax.get_ylim()[1]
ax.plot(0, y_max, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

# 调整坐标范围
ax.set_xlim(0, 110)
ax.set_ylim(0, 110)

# 显示图表
plt.show()