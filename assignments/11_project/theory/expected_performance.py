"""
Plot the early, back-of-envelope performance model from the presentation
(before we actually counted assembly lines) -- see theory/cycle_model.py for
the later, exact hand-counted version.

Reasoning:
    vmac count for an NxNx64 problem, one vmac does an 8x8x8 block:
        (1/8) * N^2 vmacs

    Expected performance:
        conversion:  O(N)
        matmul:      (1/8) * N^2 + O(1)
        total:       (1/8) * N^2 + O(N)
    Baseline (fused) performance:
        1.5 * (1/8) * N^2 + O(1)

The O(1)/O(N) constants below are illustrative placeholders (the point of
this plot is the qualitative story -- unfused starts behind but improves
relative to the fused baseline as N grows -- not exact cycle numbers).

Usage:
    python3 theory/expected_performance.py

Requires: numpy, matplotlib
"""

from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

COLOR_BASELINE = "#4f9d82"       # muted teal
COLOR_UNFUSED_TOTAL = "#c97b4a"  # muted terracotta
COLOR_CONV = "#e0b45c"           # soft gold (light fill only)
COLOR_MATMUL = "#7fa87f"         # soft green (light fill only)
FILL_ALPHA = 0.25
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"

# Illustrative O(1)/O(N) constants -- see module docstring. Chosen so the
# two curves cross at N=48: since baseline's N^2 coefficient (1.5 * 1/8) is
# fixed relative to matmul's, forcing the crossover to N=48 (right of the
# domain's midpoint) necessarily makes the split asymmetric -- a big gap
# favoring baseline at N=16 (ratio ~0.43) and a modest gap favoring unfused
# by N=64 (ratio ~1.12).
MATMUL_CONST = 48
BASELINE_CONST = 0
CONV_SLOPE = 2.0


def vmac_count(n):
    return (1 / 8) * n ** 2


def conv_cycles(n):
    return CONV_SLOPE * n


def matmul_cycles(n):
    return vmac_count(n) + MATMUL_CONST


def baseline_cycles(n):
    return 1.5 * vmac_count(n) + BASELINE_CONST


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

    ax.stackplot(n, conv, matmul, colors=[COLOR_CONV, COLOR_MATMUL], alpha=FILL_ALPHA)
    ax.plot(n, baseline, color=COLOR_BASELINE, linewidth=2.5)
    ax.plot(n, unfused_total, color=COLOR_UNFUSED_TOTAL, linewidth=2.5)

    ax.set_xlabel("N")
    ax.set_xticks([16, 32, 64])
    ax.set_yticks([])
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_TEXT_SECONDARY)

    fig.tight_layout()
    out_path = SCRIPT_DIR / "expected_performance.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
