import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

K = 64
N = np.linspace(16, 100, 400)

# expected cycles (ignoring constant warmup/cooldown), M = N
on_the_fly   = 1.5 * (N * N * K) / 512
preconverted = (N * N * K) / 512 + (3.0 / 128.0) * (N + N) * K

# Okabe-Ito colorblind-safe pair
C_OTF = "#E69F00"  # orange
C_PRE = "#0072B2"  # blue
INK   = "#222222"
MUTED = "#666666"
GRID  = "#dddddd"

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)

# region where pre-converting wins (N > 48)
ax.axvspan(48, 100, color=C_PRE, alpha=0.06, zorder=0)

ax.plot(N, on_the_fly,   color=C_OTF, lw=2.3, label="on-the-fly conversion", zorder=3)
ax.plot(N, preconverted, color=C_PRE, lw=2.3, label="pre-converted, this work", zorder=3)

# crossover point
xc, yc = 48, 1.5 * (48**2 * K) / 512
ax.axvline(xc, color=MUTED, lw=1.0, ls="--", zorder=1)
ax.plot([xc], [yc], "o", ms=7, color=INK, zorder=5,
        markeredgecolor="white", markeredgewidth=1.5)
ax.annotate("$N = 48$", xy=(xc, yc), xytext=(xc - 3, yc + 850),
            color=INK, fontsize=10, ha="right", va="bottom",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))

ax.set_xlim(16, 100)
ax.set_ylim(0, on_the_fly[-1] * 1.05)
ax.set_xlabel("$N$", fontsize=11, color=INK)
ax.set_ylabel("cycles", fontsize=11, color=INK)

ax.grid(True, axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: (f"{v/1000:g}k" if v >= 1000 else f"{v:g}")))

ax.legend(loc="upper left", frameon=False, fontsize=9.5,
          handlelength=1.6, labelcolor=INK)

fig.tight_layout()

out = os.path.join(os.path.dirname(__file__), "speedup_crossover.png")
fig.savefig(out, dpi=150)
print("wrote", out)
