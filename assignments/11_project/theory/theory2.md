# Kernel-unfusing: converting once instead of on-the-fly

<!-- Chapter outline (working draft, filled in one at a time):
1. Where we start — the fused kernel from the past assignments (single-tile, N=16, K=64, conversion+vmac interleaved)
2. The idea — unfusing conversion from compute, writing converted data back to L1 and reading from there instead of converting again
3. Expected speedup — issue-slot accounting for fused vs. unfused, accounting for size16 having no reuse and for lost load/compute interleaving
4. The actual kernels — real cycle counts and vmac fraction, measured from the assembly we wrote
5. Benchmarks — (empty for now)
-->

## 1. Where we start: the fused kernel from the past assignments

Over the past assignments we built up a single `matmul` kernel that does the full BF16 GEMM for one 16×16 output tile (`N = 16`, `K = 64`) in one pass. `A` and `B` are square in the row/column dimension we tile over (`M = N`), so we refer to that single dimension as `N` throughout: for every step of the reduction it loads a slice of `A` and `B`, converts both from BF16 to BFP16, and immediately multiply-accumulates the converted slices with `vmac.f`. Conversion and compute are *fused* into the same instruction stream, bundle by bundle. This is the kernel now sitting in `src/baseline/matmul.s`.

Concretely, the conversion path per slice is:

- `vlda.conv.fp32.bf16` — load a BF16 block from L1 and widen it to FP32 (`cml`/`cmh` registers), one 512-bit half at a time.
- `vshuffle` — rearrange the loaded FP32 lanes into the pairing that `vmul.f` expects.
- `vmul.f` — multiply by a shared power-of-two scale to bring values into the BFP16 mantissa's representable range.
- `vconv.bfp16ebs8.fp32` — pack the scaled FP32 values down into a BFP16ebs8 block (`ex` registers): 8 elements per block, 1 shared exponent byte + 8 signed mantissa bytes.

Only once both operands are staged as `ex` registers does `vmac.f dm, dm, exA, exB, r_scale` run, accumulating into the FP32 accumulator registers (`dm0`–`dm3`, one per output quadrant of the 16×16 tile).

The L1 tiling for this kernel is `p = 2, q = 2, r = 8, m = 8, n = 8, k = 8`, with views `in0: prmk`, `in1: rqkn` (B transposed), `out: pqmn` — i.e., the 16×16×64 problem is broken into a 2×2 grid of 8×8 output quadrants, reduced over 8 steps of `k = 8`.

The kernel's own header comment gives its cycle count directly: `ints = 3 + 3*6 + 2*2*6 + 4*6 + 9 = 78` cycles for the whole 16×16×64 tile — we'll use this as the real, measured fused-baseline number rather than re-deriving it, since every line in the `.s` file is one VLIW bundle (one cycle), and this kernel uses a `.rept 2` block that the comment already accounts for.

Whether this fused approach stays efficient depends entirely on how many times a given `A`-slice or `B`-slice gets reused for more than one output tile. At `N = 16` there's exactly one tile, so no such reuse is even possible yet — that becomes relevant in Chapter 3.

## 2. The idea: unfusing conversion from compute

Once `N` grows past 16, an `A`-slice or `B`-slice can be reused across more than one output tile. The fused kernel above has no way to exploit that: every reuse walks the same slice through the conversion pipeline again, because conversion and compute are interleaved bundle-by-bundle in a single instruction stream.

Our idea is to split that one kernel into two:

- **`conv(in0, in1)`** — streams all of `A` and `B` through `vlda.conv.fp32.bf16` once, converts to BFP16ebs8, and writes the result straight back into **L1**, overwriting the same buffers `in0`/`in1` already occupy (`vst.push.576.conv.bfp16ebs8.fp32` / `vst.flush.512.conv`). No `vmac` happens here at all.
- **`matmul(in0, in1, out)`** — reads the now-BFP16 buffers back in with `vlda.fill`/`vldb.fill` + `.pop.576`, and issues nothing but `vmac.f` in its compute loop. No conversion instruction appears in this kernel.

The MLIR call sequence changes from `zero(out); matmul(in0, in1, out)` to `zero(out); conv(in0, in1); matmul(in0, in1, out)` — one extra kernel call, but no extra data movement, since `conv` operates on data already sitting in L1.

## 3. Expected speed

`vmac` and the conversion step's `vmul` occupy the same vector (V) issue slot and cannot be issued in the same cycle. In the fused kernel's steady state, 4 `vmac`s are issued per 6 cycles, the remaining 2 cycles issuing `vmul` for the next block: 1.5 cycles per `vmac`.

With conversion removed, the `matmul` kernel's V-slot is never contested by `vmul`: 1 cycle per `vmac`.

The separated `conv` pass converts every element of `A` and `B` once; this cost depends on data size, not on reuse. It also no longer overlaps with `vmac`: in the fused kernel, loading and converting the next block occurred concurrently with `vmac`/`vmul` on the current one; in the unfused design, `conv` completes before any `vmac` is issued, so its cycles are not offset by any compute.

Let `n = N/16` (K fixed at 64), so there are `n²` output tile-groups. Let `V` be the number of `vmac`s per tile-group (constant), and `C` the fixed cost of converting one `A`-band and one `B`-band. Approximating a naively-scaled fused kernel as one fused call per tile-group:

```
T_fused   = n²·V·1.5
T_unfused = n·C + n²·V
```

`T_unfused` has a term linear in `n` (the one-time conversion) and a quadratic term with a smaller per-`vmac` coefficient than `T_fused`. The ratio `T_unfused / T_fused` is dominated by the linear term when `n` is small and by `1/1.5` as `n` grows. At `n = 1` (`N = 16`), there is one tile-group, so the linear conversion term is paid without any amortization, and we expect `T_unfused > T_fused`. As `n` increases, the linear term's share of the total shrinks and the quadratic terms' ratio (`1/1.5 < 1`) dominates, so we expect the gap to close and possibly invert.
