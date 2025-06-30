# ✅ 依赖导入
import numpy as np
import pandas as pd
from scipy.integrate import simpson
from numpy import log as ln
import matplotlib.pyplot as plt

# ✅ 参数设置
K_vals = np.linspace(0.8, 1.2, 5)         # 时间结构因子 K
D_vals = np.linspace(0.1, 0.5, 5)         # 扰动强度 D
sigma_vals = np.linspace(0.3, 0.7, 5)     # 谱域宽度 σ
t_vals = np.linspace(1e-3, 50, 100000)    # 时间采样
results = []

# ✅ 联合结构熵函数 H(K, D, σ)
def compute_H(K, D, sigma):
    # 结构核 p_K(t)
    p_K = K * t_vals**(K - 1) * np.exp(-t_vals)
    
    # 外部扰动项 f_D(t)
    f_D = np.exp(-D * t_vals)              # 指数扰动模型
    
    # 谱域调制项 f_sigma(t)
    f_sigma = np.exp(-t_vals**2 / (2 * sigma**2))
    
    # 联合密度
    joint = p_K * f_D * f_sigma
    joint /= simpson(joint, t_vals)  # 归一化
    
    # 结构熵计算
    H = -simpson(joint * ln(joint + 1e-20), t_vals)
    return H

# ✅ 扫描所有 (K, D, σ) 组合
for K in K_vals:
    for D in D_vals:
        for sigma in sigma_vals:
            H = compute_H(K, D, sigma)
            results.append({
                "K": K,
                "D": D,
                "σ": sigma,
                "H(K,D,σ)": H
            })

# ✅ 输出 DataFrame
df = pd.DataFrame(results)
print(df.head(10))

# ✅ 可视化结构熵在不同参数下的变化
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
subset = df[df["σ"] == 0.5]  # 固定 σ 查看 H 对 K, D 的依赖
pivot_table = subset.pivot(index="K", columns="D", values="H(K,D,σ)")
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("结构熵 H(K,D,σ) @ σ=0.5")
plt.xlabel("扰动强度 D")
plt.ylabel("结构因子 K")
plt.show()