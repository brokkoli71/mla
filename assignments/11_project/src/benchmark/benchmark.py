"""
Benchmark the size16 / size32 / size64 NPU matmul kernels, and (where built)
the fused baseline kernel for the same size.

All kernels compute out = in0 @ in1 with M=N=size, K=64. This script times
the *_benchmark.mlir kernel for each size: its inner loop repeats matmul
(and conv, for the unfused kernels) BENCH_REPS times per host launch, so a
single kernel()+wait() drives BENCH_REPS on-chip iterations and host
dispatch overhead is amortized over all of them rather than paid once per
sample. Latency is reported as a median (with p25/p75) rather than a plain
mean since a handful of host-scheduling outliers can otherwise skew a small
sample.

The "unfused" kernels transpose in1 on the host (N x K) and un-tile the
output on the host. The "baseline" kernel takes in1 in its natural K x N
layout and writes the output already row-major (DMA handles both on-device).
For size32/size64, the baseline32/baseline64 kernels reuse size16's
16x64/64x16/16x16 data movement with a padded matmul inner k-loop.

For the unfused kernels, matmul_benchmark_conv.mlir/matmul_benchmark_matmul.mlir
loop only conv or only matmul (the other runs once), isolating each kernel's
share of the total per-call time.

Usage (from the assignment directory, after building xclbins):
    python3 src/benchmark/benchmark.py

Latency and throughput (each vs. size, one line per kernel) are written
next to this script as latency_us.png and throughput_gflops.png. The
conv-vs-matmul split is written as conv_matmul_split_us.png (stacked
latency) and conv_matmul_split_gflops.png (per-kernel throughput, not
stacked since GFLOP/s isn't additive).

Requires: pyxrt, numpy, torch, matplotlib (for plots; benchmarking itself
works without it)
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import pyxrt

SIZES = (16, 32, 64)
ITERS = 300    # timed kernel launches per size
WARMUP = 20    # untimed warmup launches per size
K = 64         # inner dimension is fixed at 64 for all kernels
SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed categorical order (never cycled): slot 1 = unfused, slot 2 = baseline,
# slot 3 = conv, slot 4 = matmul.
COLOR_UNFUSED = "#2a78d6"   # blue
COLOR_BASELINE = "#1baf7a"  # aqua
COLOR_CONV = "#eda100"      # yellow
COLOR_MATMUL = "#008300"    # green
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def xclbin_paths(size: int, baseline: bool, bench: bool = False):
    name = "matmul_benchmark" if bench else "matmul"
    if baseline:
        return (
            f"build/final_{name}_baseline_size{size}.xclbin",
            f"build/insts_{name}_baseline_size{size}.bin",
        )
    return (
        f"build/final_{name}_size{size}.xclbin",
        f"build/insts_{name}_size{size}.bin",
    )


def load_kernel(size: int, baseline: bool, bench: bool = False):
    xclbin_path, insts_path = xclbin_paths(size, baseline, bench)
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


BENCH_REPS = 100000  # must match %c_reps in the matmul_benchmark.mlir files


def benchmark(size: int, iters: int, warmup: int, baseline: bool = False):
    """
    Steady-state per-call latency, measured via the *_benchmark.mlir kernel:
    its inner loop repeats matmul (and conv, for the unfused kernels)
    BENCH_REPS times inside a single acquire/release cycle, so one host
    kernel()+wait() drives BENCH_REPS on-chip iterations before returning.
    Dividing the elapsed time by BENCH_REPS amortizes host dispatch overhead
    over many iterations instead of paying it once per timed sample, and
    lets the double-buffered objectfifos overlap DMA fill/drain with compute
    across iterations the way a real back-to-back workload would.
    """
    device, kernel, bo_instr, insts = load_kernel(size, baseline, bench=True)
    _, _, _, bo_in0, bo_in1, bo_out = make_buffers(device, size, baseline)

    def launch():
        return kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)

    for _ in range(warmup):
        launch().wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch().wait()
        times.append((time.perf_counter() - t0) / BENCH_REPS)

    return _stats_from_times(times, size)


def benchmark_component(size: int, iters: int, warmup: int, component: str):
    """
    Time the conv-only or matmul-only variant of the unfused kernel
    (matmul_benchmark_conv.mlir / matmul_benchmark_matmul.mlir): one of the
    two kernels loops BENCH_REPS times while the other runs once, isolating
    each kernel's share of the total per-call time.
    """
    xclbin_path = f"build/final_matmul_benchmark_{component}_size{size}.xclbin"
    insts_path = f"build/insts_matmul_benchmark_{component}_size{size}.bin"
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

    _, _, _, bo_in0, bo_in1, bo_out = make_buffers(device, size, baseline=False)

    def launch():
        return kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)

    for _ in range(warmup):
        launch().wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch().wait()
        times.append((time.perf_counter() - t0) / BENCH_REPS)

    return _stats_from_times(times, size)


def benchmark_dummy_baseline(label_size: int, iters: int, warmup: int):
    """
    Time the baseline32/baseline64 kernels. Their data movement is fixed at
    size16's 16x64/64x16/16x16 buffers (the matmul's inner k-loop is padded
    with .rept 2 / .rept 4 instead), so buffers are allocated at size 16
    while FLOP/s is computed using label_size (32 or 64).
    """
    xclbin_path = f"build/final_matmul_benchmark_baseline_size{label_size}.xclbin"
    insts_path = f"build/insts_matmul_benchmark_baseline_size{label_size}.bin"
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

    _, _, _, bo_in0, bo_in1, bo_out = make_buffers(device, 16, baseline=True)

    def launch():
        return kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)

    for _ in range(warmup):
        launch().wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch().wait()
        times.append((time.perf_counter() - t0) / BENCH_REPS)

    return _stats_from_times(times, label_size)


def benchmark_param_baseline(size: int, iters: int, warmup: int):
    """
    Time the *real* fused baseline at size32/size64 using the parameterized
    tensor kernel (config m2=n2=2 for size32, m2=n2=4 for size64, k1=8), instead
    of the size16-padded baseline32/baseline64 approximations.

    The kernel expects tiled inputs, but timing is data-layout independent, so
    the size-matched baseline buffers from make_buffers() (32x64/64x32/32x32 or
    64x64/... -- identical byte counts to the kernel's tiled operands) are reused
    directly. Output is not numerically meaningful here (accumulator isn't
    re-zeroed across the BENCH_REPS reps), exactly as for the other benchmarks.
    """
    xclbin_path = f"build/final_tensor_kernel_param_benchmark_size{size}.xclbin"
    insts_path = f"build/insts_tensor_kernel_param_benchmark_size{size}.bin"
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

    _, _, _, bo_in0, bo_in1, bo_out = make_buffers(device, size, baseline=True)

    def launch():
        return kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)

    for _ in range(warmup):
        launch().wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch().wait()
        times.append((time.perf_counter() - t0) / BENCH_REPS)

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

    unfused_med = [v * 1e6 for v in col("unfused", "median_s")]
    unfused_p25 = [v * 1e6 for v in col("unfused", "p25_s")]
    unfused_p75 = [v * 1e6 for v in col("unfused", "p75_s")]
    baseline_med = [v * 1e6 if v is not None else None for v in col("baseline", "median_s")]
    baseline_p25 = [v * 1e6 for v in col("baseline", "p25_s", default=0.0)]
    baseline_p75 = [v * 1e6 for v in col("baseline", "p75_s", default=0.0)]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    _line_with_band(ax, x_positions, unfused_med, unfused_p25, unfused_p75,
                     "unfused", COLOR_UNFUSED)
    _line_with_band(ax, x_positions, baseline_med, baseline_p25, baseline_p75,
                     "fused (baseline)", COLOR_BASELINE)
    _configure_axes(ax, sizes, "latency (us), median with p25-p75")
    ax.set_title("NPU matmul kernel latency", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    latency_path = SCRIPT_DIR / "latency_us.png"
    fig.savefig(latency_path, dpi=150)
    plt.close(fig)

    unfused_gflops_med = col("unfused", "gflops")
    unfused_gflops_p25 = col("unfused", "gflops_p25")
    unfused_gflops_p75 = col("unfused", "gflops_p75")
    baseline_gflops_med = col("baseline", "gflops")
    baseline_gflops_p25 = col("baseline", "gflops_p25", default=0.0)
    baseline_gflops_p75 = col("baseline", "gflops_p75", default=0.0)

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


def plot_conv_matmul_split(results, sizes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed, skipping plots (pip install matplotlib)")
        return

    sizes = sorted(s for s in sizes if ("conv", s) in results and ("matmul", s) in results)
    if not sizes:
        return

    # Real numeric size on the x-axis (not evenly-spaced categorical
    # positions) -- conv scaling linearly and matmul scaling quadratically
    # with size only shows up as a visibly different slope/curvature if the
    # x-axis spacing reflects that sizes double (16, 32, 64), not just their
    # rank order.
    conv_us = [results[("conv", s)]["median_s"] * 1e6 for s in sizes]
    matmul_us = [results[("matmul", s)]["median_s"] * 1e6 for s in sizes]
    total_us = [c + m for c, m in zip(conv_us, matmul_us)]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.stackplot(sizes, conv_us, matmul_us, labels=["conv", "matmul"],
                 colors=[COLOR_CONV, COLOR_MATMUL])
    ax.plot(sizes, conv_us, color=COLOR_TEXT_PRIMARY, marker="o", markersize=6, linewidth=0)
    ax.plot(sizes, total_us, color=COLOR_TEXT_PRIMARY, marker="o", markersize=6, linewidth=0)

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"size{s}" for s in sizes], color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel("median latency (us)", color=COLOR_TEXT_PRIMARY)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=COLOR_TEXT_SECONDARY, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)

    ax.set_title("Unfused kernel: conv vs. matmul split", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    split_path = SCRIPT_DIR / "conv_matmul_split_us.png"
    fig.savefig(split_path, dpi=150)
    plt.close(fig)

    print(f"[plots] wrote {split_path}")

    # GFLOP/s isn't additive the way latency is, so conv and matmul get their
    # own lines here rather than a stacked area -- this shows each kernel's
    # own throughput, not a share of a combined total.
    conv_gflops = [results[("conv", s)]["gflops"] for s in sizes]
    matmul_gflops = [results[("matmul", s)]["gflops"] for s in sizes]
    baseline_sizes = [s for s in sizes if ("baseline", s) in results]
    baseline_gflops = [results[("baseline", s)]["gflops"] for s in baseline_sizes]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)
    ax.plot(sizes, conv_gflops, color=COLOR_CONV, linewidth=2, marker="o", markersize=8, label="conv")
    ax.plot(sizes, matmul_gflops, color=COLOR_MATMUL, linewidth=2, marker="o", markersize=8, label="matmul")
    if baseline_sizes:
        ax.plot(baseline_sizes, baseline_gflops, color=COLOR_BASELINE, linewidth=2, marker="o",
                markersize=8, label="fused (baseline)")

    ax.set_xticks(sizes)
    ax.set_xticklabels([f"size{s}" for s in sizes], color=COLOR_TEXT_PRIMARY)
    ax.set_ylabel("GFLOP/s (median)", color=COLOR_TEXT_PRIMARY)
    ax.tick_params(colors=COLOR_TEXT_SECONDARY)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_TEXT_SECONDARY)
    ax.yaxis.grid(True, color=COLOR_TEXT_SECONDARY, alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=COLOR_TEXT_PRIMARY)

    ax.set_title("conv vs. matmul vs. fused throughput", color=COLOR_TEXT_PRIMARY)
    fig.tight_layout()
    split_gflops_path = SCRIPT_DIR / "conv_matmul_split_gflops.png"
    fig.savefig(split_gflops_path, dpi=150)
    plt.close(fig)

    print(f"[plots] wrote {split_gflops_path}")


def main():
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
    for size in SIZES:
        stats = benchmark(size, ITERS, WARMUP, baseline=False)
        results[("unfused", size)] = stats
        print_row(f"unfused size{size}", stats)

        conv_path = f"build/final_matmul_benchmark_conv_size{size}.xclbin"
        if os.path.exists(conv_path):
            conv_stats = benchmark_component(size, ITERS, WARMUP, "conv")
            results[("conv", size)] = conv_stats
            print_row(f"unfused size{size} conv", conv_stats)

        matmul_path = f"build/final_matmul_benchmark_matmul_size{size}.xclbin"
        if os.path.exists(matmul_path):
            matmul_stats = benchmark_component(size, ITERS, WARMUP, "matmul")
            results[("matmul", size)] = matmul_stats
            print_row(f"unfused size{size} matmul", matmul_stats)

        # Prefer the real parameterized fused baseline (size32 -> m2=n2=2,
        # size64 -> m2=n2=4) when its benchmark xclbin is built.
        param_path = f"build/final_tensor_kernel_param_benchmark_size{size}.xclbin"
        if os.path.exists(param_path):
            base_stats = benchmark_param_baseline(size, ITERS, WARMUP)
            results[("baseline", size)] = base_stats
            speedup = base_stats["median_s"] / stats["median_s"]
            print_row(f"baseline size{size} (param)", base_stats, speedup, "speedup with unfused")
            continue

        regular_baseline_path, _ = xclbin_paths(size, baseline=True)
        if os.path.exists(regular_baseline_path):
            base_stats = benchmark(size, ITERS, WARMUP, baseline=True)
            results[("baseline", size)] = base_stats
            speedup = base_stats["median_s"] / stats["median_s"]
            print_row(f"baseline size{size}", base_stats, speedup, "speedup with unfused")
            continue

        dummy_path = f"build/final_matmul_benchmark_baseline_size{size}.xclbin"
        if os.path.exists(dummy_path):
            base_stats = benchmark_dummy_baseline(size, ITERS, WARMUP)
            results[("baseline", size)] = base_stats
            speedup = base_stats["median_s"] / stats["median_s"]
            print_row(f"baseline size{size}", base_stats, speedup, "speedup with unfused")
            continue

        print(f"{'baseline size' + str(size):>18}  (no baseline xclbin built yet, skipped)")

    plot_results(results, SIZES)
    plot_conv_matmul_split(results, SIZES)


if __name__ == "__main__":
    main()
