import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =========================
# Data
# =========================

strategies = [
    "Top 25 Ctx.",
    "Top 50 Ctx.",
    "Top 100 Ctx.",
    "Top 200 Ctx.",
    "All Ctx.",
]

ndt_inf = np.array([

    0.108,
    0.070,
    0.035,
    0.025,
    0.023,
])

dt_random = np.array([

    0.040,
    0.111,
    0.163,
    0.158,
    0.204,
])

dt_regr = np.array([

    0.023,
    0.040,
    0.048,
    0.053,
    0.053,
])

avg_random = np.array([

    2.48,
    2.89,
    3.49,
    3.58,
    4.16,
])

avg_regr = np.array([

    1.68,
    1.81,
    1.92,
    2.02,
    2.15,
])

# =========================
# Plot setup
# =========================

x = np.arange(len(strategies))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 6.5))  # keep compact height

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

for i in range(len(strategies)):
    total_random = ndt_inf[i] + dt_random[i]
    total_regr = ndt_inf[i] + dt_regr[i]

    ax.text(
        x[i] - width / 2,
        total_random + 0.01,
        f"{total_random:.3f}\n({avg_random[i]:.2f})",
        ha="center",
        va="bottom",
        fontsize=16,
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
ax.set_xticklabels(strategies, rotation=60, ha="right", fontsize=20)

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
    label=r"$DT_{>5}$ (LTLTrust Ordering)",
)

avg_handle = Line2D(
    [0], [0],
    color="black",
    linestyle="None",
    label="Top label = NDT + DT; parentheses = avg. traces"
)

ax.legend(handles=[base_patch, random_patch, regr_patch], fontsize=18)

# =========================
# Limits
# =========================

ax.set_ylim(0, np.max(ndt_inf + np.maximum(dt_random, dt_regr)) * 1.7)

plt.tight_layout()

# =========================
# Save SVG
# =========================

plt.savefig("stacked_bar_plot.svg", format="svg", bbox_inches="tight")

plt.show()