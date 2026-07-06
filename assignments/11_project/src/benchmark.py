"""
Benchmark the size16 / size32 / size64 NPU matmul kernels, and (where built)
the fused baseline kernel for the same size.

All kernels compute out = in0 @ in1 with M=N=size, K=64. This script loads
each kernel's xclbin, verifies correctness once, then times repeated kernel
launches (host-side wall clock around kernel() + wait()) and reports
median/mean latency plus achieved GFLOP/s. Latency is reported as a median
(with p10/p90) rather than a plain mean since a handful of host-scheduling
outliers can otherwise skew a small sample.

The "unfused" kernels transpose in1 on the host (N x K) and un-tile the
output on the host. The "baseline" kernel takes in1 in its natural K x N
layout and writes the output already row-major (DMA handles both on-device).
Only a size16 baseline currently exists; other sizes are skipped with a
note until their xclbins are added.

Usage (from the assignment directory, after building xclbins):
    python3 src/benchmark.py                       # benchmark size 16, 32, 64 (+ baseline where present)
    python3 src/benchmark.py --sizes 32 64          # only a subset
    python3 src/benchmark.py --iters 200 --warmup 20
    python3 src/benchmark.py --no-baseline          # skip baseline comparison

Requires: pyxrt, numpy, torch
"""

import argparse
import os
import time

import numpy as np
import torch
import pyxrt

SIZES = (16, 32, 64)
K = 64  # inner dimension is fixed at 64 for all kernels


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


def benchmark(size: int, iters: int, warmup: int, baseline: bool = False):
    device, kernel, bo_instr, insts = load_kernel(size, baseline)
    data_in0, data_in1, data_out, bo_in0, bo_in1, bo_out = make_buffers(device, size, baseline)

    # Correctness check before timing.
    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)
    out = read_out_rowmajor(bo_out, size, baseline)
    verify(data_in0, data_in1, out)

    for _ in range(warmup):
        h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
        h.wait()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
        h.wait()
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    median_s = float(np.median(times))
    mean_s = float(np.mean(times))
    p10_s, p90_s = (float(x) for x in np.percentile(times, [10, 90]))

    flops = 2 * size * size * K
    gflops = flops / median_s / 1e9

    return {
        "median_s": median_s,
        "mean_s": mean_s,
        "p10_s": p10_s,
        "p90_s": p90_s,
        "gflops": gflops,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the NPU matmul kernels")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES),
                         help="Kernel sizes to benchmark (default: 16 32 64)")
    parser.add_argument("--iters", type=int, default=100,
                         help="Timed kernel launches per size (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                         help="Untimed warmup launches per size (default: 10)")
    parser.add_argument("--no-baseline", action="store_true",
                         help="Skip baseline kernel comparison")
    args = parser.parse_args()

    header = (
        f"{'kernel':>18}  {'median (us)':>11}  {'p10-p90 (us)':>15}  "
        f"{'mean (us)':>10}  {'GFLOP/s':>9}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for size in args.sizes:
        stats = benchmark(size, args.iters, args.warmup, baseline=False)
        results[("unfused", size)] = stats
        print(
            f"{'unfused size' + str(size):>18}  {stats['median_s']*1e6:>11.2f}  "
            f"{stats['p10_s']*1e6:>6.2f}-{stats['p90_s']*1e6:<7.2f}  "
            f"{stats['mean_s']*1e6:>10.2f}  {stats['gflops']:>9.2f}"
        )

        if not args.no_baseline:
            xclbin_path, _ = xclbin_paths(size, baseline=True)
            if not os.path.exists(xclbin_path):
                print(f"{'baseline size' + str(size):>18}  (no baseline xclbin built yet, skipped)")
                continue
            base_stats = benchmark(size, args.iters, args.warmup, baseline=True)
            results[("baseline", size)] = base_stats
            speedup = base_stats["median_s"] / stats["median_s"]
            print(
                f"{'baseline size' + str(size):>18}  {base_stats['median_s']*1e6:>11.2f}  "
                f"{base_stats['p10_s']*1e6:>6.2f}-{base_stats['p90_s']*1e6:<7.2f}  "
                f"{base_stats['mean_s']*1e6:>10.2f}  {base_stats['gflops']:>9.2f}"
                f"   ({speedup:.2f}x speedup with unfused)"
            )


if __name__ == "__main__":
    main()
