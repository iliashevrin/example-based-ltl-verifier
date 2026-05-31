import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =========================
# Data
# =========================

restrictions = [
    "Top 5 Mut.",
    "Top 1 Mut.",
    "Top 1 Ctx.",
    "Only Acc.",
    "Acc. + Rej.",
    "Depth ≤ 1",
    "Depth > 1",
    "All Contexts",
]

ndt_inf = np.array([
    0.033,
    0.088,
    0.452,  # NEW
    0.309,
    0.035,
    0.063,
    0.055,
    0.020,
])

dt_random = np.array([
    0.160,
    0.092,
    0.001,  # NEW
    0.072,
    0.159,
    0.144,
    0.212,
    0.204,
])

dt_regr = np.array([
    0.103,
    0.025,
    0.000,  # NEW
    0.101,
    0.111,
    0.075,
    0.143,
    0.118,
])

avg_random = np.array([
    3.47,
    7.17,
    1.05,  # NEW
    2.42,
    3.50,
    3.37,
    4.09,
    4.32,
])

avg_regr = np.array([
    2.69,
    1.79,
    1.07,  # NEW
    2.69,
    2.88,
    2.59,
    3.42,
    3.10,
])

# =========================
# Plot setup
# =========================

x = np.arange(len(restrictions))
width = 0.38

fig, ax = plt.subplots(figsize=(16, 5))  # keep compact height

base_color = "0.80"
top_color = "0.55"

hatch_random = "//"
hatch_regr = "\\\\"

# =========================
# Bars
# =========================

ax.bar(x - width / 2, ndt_inf, width,
       color=base_color, edgecolor="black", linewidth=0.8)

ax.bar(x - width / 2, dt_random, width,
       bottom=ndt_inf,
       color=top_color, edgecolor="black",
       hatch=hatch_random, linewidth=0.8)

ax.bar(x + width / 2, ndt_inf, width,
       color=base_color, edgecolor="black", linewidth=0.8)

ax.bar(x + width / 2, dt_regr, width,
       bottom=ndt_inf,
       color=top_color, edgecolor="black",
       hatch=hatch_regr, linewidth=0.8)

# =========================
# Labels (sum + avg traces)
# =========================

for i in range(len(restrictions)):
    total_random = ndt_inf[i] + dt_random[i]
    total_regr = ndt_inf[i] + dt_regr[i]

    ax.text(
        x[i] - width / 2,
        total_random + 0.01,
        f"{total_random:.3f}\n({avg_random[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=15,
    )

    ax.text(
        x[i] + width / 2,
        total_regr + 0.01,
        f"{total_regr:.3f}\n({avg_regr[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=15,
    )

# =========================
# Formatting
# =========================

ax.set_xticks(x)
ax.set_xticklabels(restrictions, rotation=20, ha="right", fontsize=18)

ax.set_xlabel("")
ax.set_ylabel("")

# =========================
# Legend
# =========================

base_patch = mpatches.Patch(color=base_color, label=r"$NDT_\infty$")
random_patch = mpatches.Patch(
    facecolor=top_color,
    hatch=hatch_random,
    edgecolor="black",
    label=r"$DT_{>5}$ (Random)",
)
regr_patch = mpatches.Patch(
    facecolor=top_color,
    hatch=hatch_regr,
    edgecolor="black",
    label=r"$DT_{>5}$ (Regression ($\alpha=1$))",
)

avg_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="None",
    label="Top label = NDT + DT; parentheses = avg. traces"
)

ax.legend(handles=[base_patch, random_patch, regr_patch], fontsize=16)

# =========================
# Limits
# =========================

ax.set_ylim(0, np.max(ndt_inf + np.maximum(dt_random, dt_regr)) * 1.35)

plt.tight_layout()

# =========================
# Save SVG
# =========================

plt.savefig("stacked_bar_plot.svg", format="svg", bbox_inches="tight")

plt.show()