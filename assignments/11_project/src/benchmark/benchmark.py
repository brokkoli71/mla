"""
Benchmark the size16 / size32 / size64 NPU matmul kernels, and (where built)
the fused baseline kernel for the same size.

All kernels compute out = in0 @ in1 with M=N=size, K=64. This script loads
each kernel's xclbin, verifies correctness once, then times repeated kernel
launches (host-side wall clock around kernel() + wait()) and reports
median/mean latency plus achieved GFLOP/s. Latency is reported as a median
(with p25/p75) rather than a plain mean since a handful of host-scheduling
outliers can otherwise skew a small sample.

The "unfused" kernels transpose in1 on the host (N x K) and un-tile the
output on the host. The "baseline" kernel takes in1 in its natural K x N
layout and writes the output already row-major (DMA handles both on-device).
Only a size16 baseline currently exists; other sizes are skipped with a
note until their xclbins are added.

Usage (from the assignment directory, after building xclbins):
    python3 src/benchmark/benchmark.py                       # benchmark size 16, 32, 64 (+ baseline where present)
    python3 src/benchmark/benchmark.py --sizes 32 64          # only a subset
    python3 src/benchmark/benchmark.py --iters 200 --warmup 20
    python3 src/benchmark/benchmark.py --no-baseline          # skip baseline comparison
    python3 src/benchmark/benchmark.py --no-plot              # skip PNG plots

Plots (median latency, GFLOP/s) are written next to this script, in the same
directory, as latency_us.png and throughput_gflops.png.

Requires: pyxrt, numpy, torch, matplotlib (for plots; benchmarking itself
works without it)
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import pyxrt

SIZES = (16, 32, 64)
K = 64  # inner dimension is fixed at 64 for all kernels
SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed categorical order (never cycled): slot 1 = unfused, slot 2 = baseline.
COLOR_UNFUSED = "#2a78d6"   # blue
COLOR_BASELINE = "#1baf7a"  # aqua
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def xclbin_paths(size: int, baseline: bool):
    if baseline:
        return (
            f"build/final_matmul_baseline_size{size}.xclbin",
            f"build/insts_matmul_baseline_size{size}.bin",
        )
    return (
        f"build/final_matmul_size{size}.xclbin",
        f"build/insts_matmul_size{size}.bin",
    )


def load_kernel(size: int, baseline: bool):
    xclbin_path, insts_path = xclbin_paths(size, baseline)
    insts = np.fromfile(insts_path, dtype=np.uint32)

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    context = pyxrt.hw_context(device, uuid)
    kname = xclbin.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(context, kname)

    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

    return device, kernel, bo_instr, insts


def make_buffers(device, size: int, baseline: bool):
    torch.manual_seed(42)
    data_in0 = torch.randn(size, K, dtype=torch.bfloat16)
    data_in1 = torch.randn(K, size, dtype=torch.bfloat16)   # natural K x N layout
    data_out = torch.zeros(size, size, dtype=torch.bfloat16)

    # The baseline kernel consumes in1 directly in K x N layout; the unfused
    # kernels require it pre-transposed to N x K on the host.
    data_in1_kernel = data_in1 if baseline else data_in1.t().contiguous()

    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1_kernel.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1_kernel.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1_kernel.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    return data_in0, data_in1, data_out, bo_in0, bo_in1, bo_out


def read_out_rowmajor(bo_out, size: int, baseline: bool) -> torch.Tensor:
    tensor_out = torch.frombuffer(
        bo_out.map(), dtype=torch.bfloat16, count=size * size
    ).view(size, size)

    # The baseline kernel's output DMA already reorders to row-major on-chip.
    # The unfused kernels write output tiles in [p_hi][q_hi][p_lo][q_lo][m][n]
    # order (p_lo/q_lo = 2, m/n = 8 core tiles); size16 has a single p_hi/q_hi
    # tile so it's already row-major there too.
    tiles = size // 16
    if baseline or tiles == 1:
        return tensor_out.clone()

    return (
        tensor_out.reshape(tiles, tiles, 2, 2, 8, 8)
        .permute(0, 2, 4, 1, 3, 5)      # -> [p_hi][p_lo][m][q_hi][q_lo][n]
        .reshape(size, size)
        .contiguous()
    )


def verify(in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    ref = in0 @ in1
    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.02)


def _stats_from_times(times, size: int):
    times = np.array(times)
    median_s = float(np.median(times))
    mean_s = float(np.mean(times))
    p25_s, p75_s = (float(x) for x in np.percentile(times, [25, 75]))

    flops = 2 * size * size * K
    gflops = flops / median_s / 1e9
    gflops_samples = flops / times / 1e9
    gflops_p25, gflops_p75 = (float(x) for x in np.percentile(gflops_samples, [25, 75]))

    return {
        "median_s": median_s,
        "mean_s": mean_s,
        "p25_s": p25_s,
        "p75_s": p75_s,
        "gflops": gflops,
        "gflops_p25": gflops_p25,
        "gflops_p75": gflops_p75,
    }


def _setup_and_verify(size: int, baseline: bool):
    """Load the kernel, set up buffers, and check correctness with one call."""
    device, kernel, bo_instr, insts = load_kernel(size, baseline)
    data_in0, data_in1, data_out, bo_in0, bo_in1, bo_out = make_buffers(device, size, baseline)

    def launch():
        return kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)

    h = launch()
    h.wait()
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)
    out = read_out_rowmajor(bo_out, size, baseline)
    verify(data_in0, data_in1, out)

    return launch


def benchmark(size: int, iters: int, warmup: int, baseline: bool = False):
    """
    Serialized round-trip latency: launch one kernel invocation, wait for it
    to fully complete, then launch the next. Each timed sample includes host
    dispatch overhead, on-chip DMA fill/drain, and compute for that one call.
    """
    launch = _setup_and_verify(size, baseline)

    for _ in range(warmup):
        launch().wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch().wait()
        times.append(time.perf_counter() - t0)

    return _stats_from_times(times, size)


def _line_with_band(ax, x_positions, center, lower, upper, label, color):
    # Percentile band (p25-p75) rather than mean +/- 1 std: these per-call
    # host latencies are right-skewed (occasional slow dispatch/scheduling
    # outliers pull the mean above the median), so a symmetric std band
    # around the mean would misrepresent the spread. Median + percentiles is
    # robust to that skew.
    xs = [xi for xi, c in zip(x_positions, center) if c is not None]
    ys = [c for c in center if c is not None]
    los = [lo for lo, c in zip(lower, center) if c is not None]
    his = [hi for hi, c in zip(upper, center) if c is not None]
    if not xs:
        return

    if len(xs) >= 2:
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=label)
        ax.fill_between(xs, los, his, color=color, alpha=0.18, linewidth=0)
    else:
        yerr = [[ys[0] - los[0]], [his[0] - ys[0]]]
        ax.errorbar(xs, ys, yerr=yerr, color=color, marker="o", markersize=8,
                    linewidth=0, elinewidth=2, capsize=4, label=label)


def _configure_axes(ax, sizes, ylabel):
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"size{s}" for s in sizes], color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel(ylabel, color=COLOR_TEXT_PRIMARY)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=COLOR_TEXT_SECONDARY, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)


def plot_results(results, sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed, skipping plots (pip install matplotlib)")
        return

    x_positions = list(range(len(sizes)))

    def col(kind, key, default=None):
        return [
            results[(kind, s)][key] if (kind, s) in results else default
            for s in sizes
        ]

    unfused_latency_med = [v * 1e6 for v in col("unfused", "median_s")]
    unfused_latency_p25 = [v * 1e6 for v in col("unfused", "p25_s")]
    unfused_latency_p75 = [v * 1e6 for v in col("unfused", "p75_s")]
    baseline_latency_med = col("baseline", "median_s")
    baseline_latency_med = [v * 1e6 if v is not None else None for v in baseline_latency_med]
    baseline_latency_p25 = [v * 1e6 for v in col("baseline", "p25_s", default=0.0)]
    baseline_latency_p75 = [v * 1e6 for v in col("baseline", "p75_s", default=0.0)]

    unfused_gflops_med = col("unfused", "gflops")
    unfused_gflops_p25 = col("unfused", "gflops_p25")
    unfused_gflops_p75 = col("unfused", "gflops_p75")
    baseline_gflops_med = col("baseline", "gflops")
    baseline_gflops_p25 = col("baseline", "gflops_p25", default=0.0)
    baseline_gflops_p75 = col("baseline", "gflops_p75", default=0.0)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    _line_with_band(ax, x_positions, unfused_latency_med, unfused_latency_p25, unfused_latency_p75,
                     "unfused", COLOR_UNFUSED)
    _line_with_band(ax, x_positions, baseline_latency_med, baseline_latency_p25, baseline_latency_p75,
                     "fused (baseline)", COLOR_BASELINE)
    _configure_axes(ax, sizes, "latency (us), median with p25-p75")
    ax.set_title("NPU matmul kernel latency", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    latency_path = SCRIPT_DIR / "latency_us.png"
    fig.savefig(latency_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    _line_with_band(ax, x_positions, unfused_gflops_med, unfused_gflops_p25, unfused_gflops_p75,
                     "unfused", COLOR_UNFUSED)
    _line_with_band(ax, x_positions, baseline_gflops_med, baseline_gflops_p25, baseline_gflops_p75,
                     "fused (baseline)", COLOR_BASELINE)
    _configure_axes(ax, sizes, "throughput (GFLOP/s), median with p25-p75")
    ax.set_title("NPU matmul kernel throughput", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    throughput_path = SCRIPT_DIR / "throughput_gflops.png"
    fig.savefig(throughput_path, dpi=150)
    plt.close(fig)

    print(f"[plots] wrote {latency_path}")
    print(f"[plots] wrote {throughput_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the NPU matmul kernels")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                         help="Kernel sizes to benchmark (default: 16 32 64)")
    parser.add_argument("--iters", type=int, default=300,
                         help="Timed kernel launches per size (default: 300)")
    parser.add_argument("--warmup", type=int, default=20,
                         help="Untimed warmup launches per size (default: 20)")
    parser.add_argument("--no-baseline", action="store_true",
                         help="Skip baseline kernel comparison")
    parser.add_argument("--no-plot", action="store_true",
                         help="Skip writing PNG plots")
    args = parser.parse_args()

    header = (
        f"{'kernel':>18}  {'median (us)':>11}  {'p25-p75 (us)':>15}  "
        f"{'mean (us)':>10}  {'GFLOP/s':>9}"
    )

    def print_row(label, stats, speedup=None, speedup_label=""):
        suffix = f"   ({speedup:.2f}x {speedup_label})" if speedup is not None else ""
        print(
            f"{label:>18}  {stats['median_s']*1e6:>11.2f}  "
            f"{stats['p25_s']*1e6:>6.2f}-{stats['p75_s']*1e6:<7.2f}  "
            f"{stats['mean_s']*1e6:>10.2f}  {stats['gflops']:>9.2f}{suffix}"
        )

    print(header)
    print("-" * len(header))

    results = {}
    for size in args.sizes:
        stats = benchmark(size, args.iters, args.warmup, baseline=False)
        results[("unfused", size)] = stats
        print_row(f"unfused size{size}", stats)

        if not args.no_baseline:
            xclbin_path, _ = xclbin_paths(size, baseline=True)
            if not os.path.exists(xclbin_path):
                print(f"{'baseline size' + str(size):>18}  (no baseline xclbin built yet, skipped)")
                continue
            base_stats = benchmark(size, args.iters, args.warmup, baseline=True)
            results[("baseline", size)] = base_stats
            speedup = base_stats["median_s"] / stats["median_s"]
            print_row(f"baseline size{size}", base_stats, speedup, "speedup with unfused")

    if not args.no_plot:
        plot_results(results, args.sizes)


if __name__ == "__main__":
    main()
