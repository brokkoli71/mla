"""
Plot the theoretical (hand-counted) cycle-count model for the fused baseline
kernel vs. our unfused conv+matmul kernels, as a function of N (M=N, K=64).

Cycle counts (one VLIW bundle = one cycle), from theory/berechnung.md:
    baseline (fused):      30 + (3/16) * N^2
    unfused conv:          10 + 3 * N
    unfused matmul:        24 + (1/8) * N^2

The unfused conv+matmul are plotted as a stacked area (conv on the bottom,
matmul on top) so the top edge of the stack -- the unfused total -- can be
compared directly against the fused baseline line. N runs continuously from
16 to 64 since these are closed-form formulas, not discrete measurements.

Usage:
    python3 theory/cycle_model.py

Requires: numpy, matplotlib
"""

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Muted fills for the conv/matmul background (de-emphasized: they're context,
# not the comparison); the baseline and unfused-total lines carry the actual
# comparison and stay more saturated so they read clearly on top.
COLOR_BASELINE = "#4f9d82"      # muted teal
COLOR_UNFUSED_TOTAL = "#c97b4a"  # muted terracotta
COLOR_CONV = "#e0b45c"          # soft gold (light fill only)
COLOR_MATMUL = "#7fa87f"        # soft green (light fill only)
FILL_ALPHA = 0.25
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def baseline_cycles(n):
    return 30 + (3 / 16) * n ** 2


def conv_cycles(n):
    return 10 + 3 * n


def matmul_cycles(n):
    return 24 + (1 / 8) * n ** 2


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed (pip install matplotlib)")
        return

    n = np.linspace(16, 64, 200)
    baseline = baseline_cycles(n)
    conv = conv_cycles(n)
    matmul = matmul_cycles(n)
    unfused_total = conv + matmul

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.stackplot(n, conv, matmul, labels=["conv", "matmul"],
                 colors=[COLOR_CONV, COLOR_MATMUL], alpha=FILL_ALPHA)
    ax.plot(n, baseline, color=COLOR_BASELINE, linewidth=2.5, label="fused (baseline)")
    ax.plot(n, unfused_total, color=COLOR_UNFUSED_TOTAL, linewidth=2.5,
            label="unfused total (conv + matmul)")

    for x in (16, 32, 64):
        ax.axvline(x, color=COLOR_TEXT_SECONDARY, alpha=0.15, linewidth=1)

    ax.set_xticks([16, 32, 64])
    ax.set_xticklabels(["16", "32", "64"], color=COLOR_TEXT_PRIMARY)
    ax.set_xlabel("N (M = N, K = 64)", color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel("cycles", color=COLOR_TEXT_PRIMARY)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=COLOR_TEXT_SECONDARY, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)

    ax.set_title("Cycle count: fused vs. unfused",
                 color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    out_path = SCRIPT_DIR / "cycle_model.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"wrote {out_path}")
    for size in (16, 32, 64):
        b = baseline_cycles(size)
        c = conv_cycles(size)
        m = matmul_cycles(size)
        print(f"N={size:>3}: baseline={b:>8.1f}  conv={c:>7.1f}  matmul={m:>7.1f}  "
              f"unfused total={c + m:>8.1f}")


if __name__ == "__main__":
    main()
