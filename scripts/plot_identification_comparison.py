#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制辨识结果对比条形图：验证集 RMSE 与 最大绝对误差。
支持中文显示。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 支持中文：优先 Noto Sans CJK / SimHei / 文泉驿
_cjk = [f.name for f in fm.fontManager.ttflist if ("Noto Sans CJK" in f.name or "Noto Serif CJK" in f.name or "SimHei" in f.name or "WenQuanYi" in f.name)]
plt.rcParams["font.sans-serif"] = (_cjk[:1] if _cjk else []) + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

labels = [
    "全辨识\n(URDF[COM]+摩擦)",
    "原URDF+RNEA\n(无辨识)",
    "只辨识\n质量+摩擦",
    "只辨识\n摩擦力",
    "仿真数据辨识\n(验证程序)",
]
rmse = [1.2559, 1.3555, 1.2866, 1.3158, 0.0075]  # Nm
max_err = [7.4530, 8.4962, 7.5215, 8.2738, 0.0213]  # Nm

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, rmse, width, label="验证集 RMSE (Nm)", color="steelblue")
bars2 = ax.bar(x + width / 2, max_err, width, label="验证集 最大绝对误差 (Nm)", color="coral")

ax.set_ylabel("误差 (Nm)")
ax.set_title("动力学辨识结果对比")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 仿真数据那一组数值很小，用对数刻度会更清晰；这里保持线性，在图上标注数值
def autolabel(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            f"{h:.2f}" if h >= 0.1 else f"{h:.4f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )

autolabel(bars1)
autolabel(bars2)

fig.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), "identification_comparison.png")
plt.savefig(out, dpi=150)
print(f"已保存: {out}")
try:
    plt.show()
except Exception:
    pass
