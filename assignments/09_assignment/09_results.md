# Submission 09: XDNA GEMM

This week you will perform a larger matrix multiplication on the NPU.
For this you will write data-movement code in the MLIR-AIE dialect and add loops around your XDNA tensor kernel.

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

1. `aie.tile()`: This operation creates an AIE tile in the AIE array
2. `aie.core()`: This operation represents an AIEngine processor core belonging to a tile
3. `aie.runtime_sequence()`: Instructions in this operation allow for runtime (re-)configuration of the AI Engine array, such as configuring data movement buffer descriptors. (e.g. data transfers between host and AIE array on the shims)
4. `aie.objectfifo()`: Creates a circular buffer or channel between two tiles
5. `aie.objectfifo.link()`: Links two objectFifos through an intermediary tile’s DMA (Direct Memory Access)
6. `aie.objectfifo.acquire()` and `aie.objectfifo.release()`: Acquire/Release operation to lock and return objects of an ObjectFifo
7. `aiex.npu.dma_memcpy_nd()`: An n-dimensional half DMA operator. Programs a DMA to access a memory memref with an access pattern specified by offsets, sizes and strides or static_offsets, static_sizes and static_strides
8. `aiex.npu.dma_wait()`: blocks until the DMA referenced through symbol completes execution and issues a task-complete-token. e.g. `aiex.npu.dma_wait { symbol = @out0 }`

## Task 2 - Data Layouts and Loops

**Sketch** the data movement between main memory, shim tile, memory tile, and compute tile.
- data will be loaded from main memory to the shim tile, then to the memory tile, and finally to the compute tile where the matrix multiplication will be performed. The output will be moved back in the reverse order.

**Describe** which *mlir-aie* operation is involved in each step.
- `aie.objectfifo` will create a channel from shim to memory tile and a channel from memory tile to compute tile. both will be linked by `aie.objectfifo.link`
- `aiex.npu.dma_memcpy_nd()` maps main memory to shim and `aiex.npu.dma_wait` will wait until writing is finished

**Data layout:**
- in main memory the matrices are row-major (`in0: MK`, `in1: KN`, `out: MN`). the `aiex.npu.dma_memcpy_nd` on the shim does the tiling, its offsets/sizes/strides read that row-major data as the tiled views `in0: apmcrk`, `in1: crkbqn`, `out: apmbqn`
- the actual layout change to the L1 layout (`in0: prmk`, `in1: rqkn`, `out: pqmn`) is done by the `dimensionsToStream` on the memory tile objectfifo (L2->L1), it reorders the dimensions while streaming into L1 so the compute kernel gets the `2x8x8x8` / `8x2x8x8` / `2x2x8x8` layout it expects.
- on the way back the same thing happens in reverse, the out objectfifo `dimensionsToStream` turns `pqmn` back into a matrix tile and the output `dma_memcpy_nd` writes it to the row-major `out: MN`.

## Task 3 - Implementation

1. **Implement** the data movement inside `src/matmul.mlir` and change the dimension sizes in `src/driver.py`.
   **Replace** the *TODOs* with the corresponding code.

2. **Verify** your implementation by executing `make run_matmul`.

Implemented in `src/matmul_task3.mlir` as a `b`-outer loop (8 blocks). Per block we send one `out` (all 16 `a`-tiles of column `b`), the full `in0`, and `in1`. Since `in1` does not depend on `a`, its leftmost DMA dimension has `size=16, stride=0` to resend it 16× (once per `a`), with the `b` column-offset in the stride-1 slot.

Can be run with `make run_matmul_task_3`
## Task 4 - Performance

**Change** the data movement inside the MLIR code so that there is no blocking wait, i.e., there is always a data movement operation that can be issued (except for the last one).

Implemented in `src/matmul.mlir`. The scheduling changed: two BD-id sets `{0,1,2}` and `{8,9,10}` alternate per block, and blocks 0 and 1 are both issued before the first `dma_wait`. Each following block is issued before waiting on the previous one.

## Task 5 - Buffer Placement (optional)

**Find** the buffer placement operations inside the lowered MLIR code (`build/matmul.mlir.prj/input_with_addresses.mlir`).
**Describe** how you would place the buffers to reduce bank conflicts and what changes would be needed to your XDNA tensor kernel.

