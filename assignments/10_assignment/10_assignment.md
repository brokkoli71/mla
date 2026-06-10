# Assignment 10: Using the whole NPU

This week you will perform the matrix multiplication on the whole NPU.
For this, you will adapt the data-movement code and execute your XDNA tensor kernel on all compute tiles.

**Prerequisites:** Run the following to enter the development environment and install torch.

```bash
nix develop nix-amd-npu#iron-full
iron-fhs
pip install torch                  # only once
```

## Data Layout and Data Movement

In main memory, the matrices are stored in row-major order (`in0: MK`, `in1: KN`, and `out: MN` with `M=256`, `N=128`, and `K=1024`).
During the data movement from L3 (main memory) to L1 (scratchpad), the matrices are first tiled and then broadcast along the columns or rows.
The dimensions are split as follows:

- `M->axpm` with `a=2`, `x=8`, `p=2`, `m=8`,
- `N->byqn` with `b=2`, `y=4`, `q=2`, `n=8`, and
- `K->crk` with `c=16`, `r=8`, `k=8`.

This yields the views `in0: axpmcrk`, `in1: crkbyqn`, and `out: axpmbyqn`.
During the data movement to the L1 scratchpad memory, the layout must be changed to `in0: prmk`, `in1: rqkn`, and `out: pqmn`.
The dimensions `a`, `b`, and `c` are handled sequentially through loops on the compute tiles.
The dimensions `x` and `y` are handled spatially by distributing them across the compute-tile columns and rows, respectively.
The DMAs move the corresponding tiles from the memory tile to the compute tiles, broadcasting the `in0` tiles along the columns and the `in1` tiles along the rows, so that each column receives a different `in0` tile and each row receives a different `in1` tile.
Before the `c` loop, the output tile is zero-initialized.
When writing the `out` tiles, the four `out` tiles produced by the compute tiles of one column are joined into the intermediate layout `ypqmn`.
The dimension `y` is realized by giving each row's L1L2 FIFO a different write offset into the joined L2 buffer.
Note that the layout is changed to `ypmqn` when reading the data from L2 to the stream.
When writing the output tiles from the memory tile to the main memory, the layout is changed to a matrix layout (`out: MN`).

## Task 1 - Setup of the Whole NPU

**Create** variables for each tile (shim, memory, and compute).
**Adapt** the fused `ab`-loop size and **duplicate** the core function for each compute tile.
**Set** the FIFO queue suffixes inside each core function so that they match its column and row: replace the placeholder suffixes `_0` (for `in0`/`in1`) and `_0_0` (for `out`) with the actual tile coordinates, yielding `in0_<col>`, `in1_<row>`, and `out_<col>_<row>`.
Row indices in the FIFO suffixes are zero-based and counted from the first compute-tile row (so `aie.tile(_, 2)` → row index 0, `aie.tile(_, 3)` → row index 1, etc.). For example, for `aie.tile(7, 3)` the suffixes are `@in0_L2L1_7`, `@in1_L2L1_1`, and `@out_L1L2_7_1`.

## Task 2 - Broadcasting the Inputs

**Extend** the existing `L2L1` input FIFO queues by adding the additional consumer tiles along row 0 (`in0`) or column 0 (`in1`).
**Create** the needed `L2L1` and `L3L2` input FIFO queues for the remaining columns and rows.
Note that only four `in1_L3L2` FIFO queues are needed. Their placement on the shim tiles can be freely chosen.
**Adapt** the input `dma_memcpy_nd` operations to their corresponding FIFO queue.
Note that each shim tile has 16 buffer descriptors; consequently, FIFO queues on *different* shim tiles may reuse the same BD id.

## Task 3 - Writing the Output

**Create** the needed `L1L2` and `L2L3` output FIFO queues and **join** the `L1L2` FIFO queues along a column to the corresponding `L2L3` FIFO queue.
**Adapt** the output `dma_memcpy_nd` operations to their corresponding FIFO queue and **change** their sizes, strides, and offsets.
Note that the `dma_memcpy_nd` operations in each shim tile receive the output tensor tiles in the layout `aby(p*m)(q*n)`, and `x` is handled spatially across the columns and therefore does not appear in this view.
The dimensions `b` and `y` could be fused in this view, but we keep them separate.

## Task 4 - Testing

**Test** your implementation by running `make run_matmul`.

## Task 5 - Selection of Spatial Dimensions (optional)

**Explain** how the data transfer would change if the roles of the dimensions `y` and `x` were switched, i.e., the dimensions were split as follows:

- `M->aypm` with `a=4`, `y=4`, `p=2`, `m=8`,
- `N->bxqn` with `b=1`, `x=8`, `q=2`, `n=8`, and
- `K->crk` with `c=16`, `r=8`, `k=8`.

In which cases should `in0` be broadcast along the rows, and in which cases should `in1` be broadcast along the rows?
When is no difference in performance expected?

## Group Specific Component

At the end of the course, your group will work on a project of your own choosing.

**Prepare** a pitch for at least two possible project ideas: one targeting the XDNA NPU and one targeting GPUs with cuTile.
**Create** at least two slides per project idea, covering

- a brief introduction,
- the problem formulation,
- the proposed solution, and
- the expected results or insights.

You will present both project ideas on 17.06.2026, with about 5 minutes for each idea.

**You do not have to hand in the slides. Hand in a short description (2-3 sentences) about each project idea.**
