# Assignment 09: XDNA GEMM

This week you will perform a larger matrix multiplication on the NPU.
For this you will write data-movement code in the MLIR-AIE dialect and add loops around your XDNA tensor kernel.

**Prerequisites:** Run the following to enter the development environment and install torch.

```bash
nix develop nix-amd-npu#iron-full
iron-fhs
pip install torch                  # only once
```

## Data Layout and Data Movement

In the main memory, the matrices are stored in row-major order (`in0: MK`, `in1: KN`, and `out: MN` with `M=256`, `N=128`, and `K=1024`).
During the data movement from L3 (main memory) to L1 (scratchpad), the matrices are tiled.
The dimensions are split as follows:

- `M->apm` with `a=16`, `p=2`, `m=8`,
- `N->bqn` with `b=8`, `q=2`, `n=8`, and
- `K->crk` with `c=16`, `r=8`, `k=8`.

This yields the views `in0: apmcrk`, `in1: crkbqn`, and `out: apmbqn`.
During the data movement to the L1 scratchpad memory, the layout must be changed to `in0: prmk`, `in1: rqkn`, and `out: pqmn`.
The dimensions `a`, `b`, and `c` are handled sequentially through loops on the compute tile. The DMAs move the corresponding tiles to the compute tile.
Before the `c` loop, the output tile is zero-initialized.
When writing the output tensor, its layout is changed back to a matrix layout (`out: MN`).

## Task 0 - Setup

**Copy** your XDNA tensor kernel into the `src/` directory and copy your `verify()` function to the driver.
Set the maximum absolute error to `2` and the maximum relative error to `0.5`.

## Task 1 - MLIR-AIE operations

**Give a brief summary** of the following *mlir-aie* operations:

1. `aie.tile()`
2. `aie.core()`
3. `aie.runtime_sequence()`
4. `aie.objectfifo()`
5. `aie.objectfifo.link()`
6. `aie.objectfifo.acquire()` and `aie.objectfifo.release()`
7. `aiex.npu.dma_memcpy_nd()`
8. `aiex.npu.dma_wait()`

## Task 2 - Data Layouts and Loops

**Sketch** the data movement between main memory, shim tile, memory tile, and compute tile.
**Describe** which *mlir-aie* operation is involved in each step.

## Task 3 - Implementation

1. **Implement** the data movement inside `src/matmul.mlir` and change the dimension sizes in `src/driver.py`.
   **Replace** the *TODOs* with the corresponding code.

2. **Verify** your implementation by executing `make run_matmul`.

## Task 4 - Performance

**Change** the data movement inside the MLIR code so that there is no blocking wait, i.e., there is always a data movement operation that can be issued (except for the last one).

## Task 5 - Buffer Placement (optional)

**Find** the buffer placement operations inside the lowered MLIR code (`build/matmul.mlir.prj/input_with_addresses.mlir`).
**Describe** how you would place the buffers to reduce bank conflicts and what changes would be needed to your XDNA tensor kernel.

