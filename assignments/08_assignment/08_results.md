# Submission 08: XDNA GEMM Kernel

This week you will write an XDNA kernel to perform a matrix multiplication on the NPU.
The XDNA kernel is a tensor kernel as it operates on tensor layouts.

## Data Layout and Data Movement

In the main memory the matrices are stored in row-major order (`in0: MK`, `in1: KN`, and `out: MN`).
During the data movement from L3 (main memory) to L1 (scratchpad), the matrices are tiled.
The dimensions are split as follows:

- `M->pm` with `p=2`, `m=8`,
- `N->qn` with `q=2`, `n=8`, and
- `K->rk` with `r=8`, `k=8`.

This yields the views `in0: pmrk`, `in1: rkqn`, and `out: pmqn`.
During the data movement to the L1 scratchpad memory, the layout is changed to `in0: prmk`, `in1: rqkn`, and `out: pqmn`.
The NPU scratchpad memory is zero-initialized during NPU setup, so within the tensor kernel you may assume that the output tensor memory is already zero.
When writing the output tensor, its layout is changed back to a matrix layout (`out: MN`).

## Task 1 — Verify Function ✅

**Implement** the `verify()` function for the matrix multiplication in `src/driver.py`.

## Task 2 — Instructions and Latencies

**Fill in** the table below with the instructions you will need for your tensor kernel.

| Instruction           | Slot | Latency |
|-----------------------|------|---------|
|  vmul.f               |  V   |  6      |
|  mov                  |  M   |  1      |
|  padds                |  S   |  1      |
|  vlda.conv.fp32.bf16  |  A   |  7      |
|  vldb                 |  A   |  6      |
|  vshuffle             |  M   |  2      |
|  vconv.bfp16ebs8.fp32 |  M   |  2      |
|  vst.conv.bf16.fp32   |  S   |  2      |
|  vbcst                |  M   |  1      |

vconv.fp32.bf16 cml1, x2 | same Slot as vshuffle M | 2
vconv.bfp16ebs8.fp32 ex7 | not the same Slot as vshuffle S | 4
vmac.f dm2, dm2, ex10, ex11, r3 | V | 6, 4 für akkumulator late forwarding

## Task 3 — Register Blocking

**Choose** a register blocking for your tensor kernel.
Assign the input and output tensors to the registers and explain your decision.
Keep in mind that the input tensors must be converted from BF16 to BFP16 (`bfp16ebs8`).

*Note: The same register may be reused for multiple tensors when their lifetimes do not overlap.*

| Tensor | Registers           |
|--------|---------------------|
| `out`  | dm1 - dm4           |
| `in0`  | dm0, ex10, ex11     |
| `in1`  | x0-x7, dm0, ex0, ex1|

The idea was to hold all results permanently in the Accumulator Registers. but that would require them to occupy dm1-dm4 (one per resulting tile), leaving only one Accumulator Register dm0 for the conversion of the inputs `bf16 -> fp32 -> bfp16`. 

## Task 4 — Data Layouts and Pointer Updates
**Sketch** the data layout and the required pointer updates corresponding to your register blocking.


In one iteration of accumulating the output we load 2 tiles of in0 and 2 of in1, multiply the tiles and add them onto the output: 
![data layout](data_layout_3.png)

### Pointers used

| Ptr | Role | Update |
|-----|------|--------|
| `p0` | `in0` base, never modified | — (only read by `mov p4, p0`) |
| `p1` | `in1` cursor, walks the `r` axis | `padds [p1], #256` once per `r`-block |
| `p2` | `out` base; store cursor in the epilogue | `[p2], #64` post-inc ×8 |
| `p3` | `out` load cursor (prologue only) | `[p3], #64` post-inc ×8 |
| `p4` | `in0` cursor, walks `p` then `r` | see below |

### `in0` — `p4` per `r`-block

Both `p` tiles are needed each iteration, so `p4` walks the `p` axis inside the block
and is then rewound by 1024−128 to land on the next `r`:

Net per block: +1024 − 896 = +128 B = one `r` step, and `p4` is back at `p=0`.
The rewind is split into three `padds` because each is co-issued for free in the S slot
of the three `vmac` bundles (and keeps each immediate in range). In the `r=0` block the
rewind is instead done as `mov p4, p0; padds [p4], #128`, since `p4` was still being set
up there.

### `in1` — `p1` per `r`-block

`p1` needs only a single `+256` per iteration and the `q` split is a constant
immediate offset.

### `out` — `p3` in, `p2` out

`out` is `pqmn` and the register blocking assigns one accumulator per `(p,q)` tile, so
both the initial load and the final store are a single linear sweep with a constant
`#64` post-increment — no strided update is needed at all:


## Task 5 — Implementation

1. **Implement** the tensor kernel in `src/matmul.s`.
   Do not use any control-flow instruction other than the final `ret lr`.

2. **Verify** your kernel with:

```bash
make run_matmul
```

## Task 6 — Performance

1. **Count** your instructions. What performance should your kernel achieve?
 Total	327 cycles, 100 FLOP/cycle
 Peak is one vmac per cycle = 1024 FLOP/cycle, i.e. a 32-cycle lower bound → the kernel runs at ≈ 9.8 % of peak, 10.2× off.
2. **Argue** whether your instruction count is minimal, or describe which optimizations could further reduce it.

Instruction count is not mimimal, further use of parallelization through using more units.

