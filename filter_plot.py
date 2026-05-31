import numpy as np
import matplotlib.pyplot as plt

# =========================
# Data (from the LaTeX table)
# =========================

restrictions = [
    "Top 5 Mut.",
    "Top 1 Mut.",
    "Only Acc.",
    "Acc. + Rej.",
    "Depth ≤ 1",
    "Depth > 1",
    "No Filter",
]

# NDT_infty (same base for both bars)
ndt_inf = np.array([
    0.033,
    0.088,
    0.309,
    0.035,
    0.063,
    0.055,
    0.020,
])

# DT_>5
dt_random = np.array([
    0.160,
    0.092,
    0.072,
    0.159,
    0.144,
    0.212,
    0.204,
])

dt_regr = np.array([
    0.103,
    0.025,
    0.101,
    0.111,
    0.075,
    0.143,
    0.118,
])

# =========================
# Plot
# =========================

x = np.arange(len(restrictions))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

# Colors
ndt_color = "lightgray"
random_color = "tab:blue"
regr_color = "tab:orange"

# Random ordering
ax.bar(
    x - width / 2,
    ndt_inf,
    width,
    color=ndt_color,
    label=r"$NDT_\infty$",
)

ax.bar(
    x - width / 2,
    dt_random,
    width,
    bottom=ndt_inf,
    color=random_color,
    label=r"$DT_{>5}$ (Random)",
)

# Regression ordering
ax.bar(
    x + width / 2,
    ndt_inf,
    width,
    color=ndt_color,
)

ax.bar(
    x + width / 2,
    dt_regr,
    width,
    bottom=ndt_inf,
    color=regr_color,
    label=r"$DT_{>5}$ (Regr.)",
)

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(restrictions, rotation=20, ha="right")

# No axis labels
ax.set_xlabel("")
ax.set_ylabel("")

ax.set_title(r"$NDT_\infty$ and $DT_{>5}$ by Restriction")

ax.legend()

ax.set_ylim(0, max(ndt_inf + np.maximum(dt_random, dt_regr)) * 1.15)

plt.tight_layout()
plt.show()