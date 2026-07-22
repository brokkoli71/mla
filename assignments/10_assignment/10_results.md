# Submission 10: Using the whole NPU
This week you will perform the matrix multiplication on the whole NPU.
For this, you will adapt the data-movement code and execute your XDNA tensor kernel on all compute tiles.

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

One `core` per compute tile (8 columns × 4 rows), FIFO suffixes set per tile: `in0_<col>`, `in1_<row>`, `out_<col>_<row>`. Tile (7,3) → col 7, row 1:
```c++
%core_7_3 = aie.core(%tile_7_3) {
  %buffer_out = aie.objectfifo.acquire @out_L1L2_7_1(Produce, 1) : ...
  %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : ...
  %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : ...
```
The 32 cores are generated with `generate.py`.

## Task 2 - Broadcasting the Inputs

`in0_L2L1` consumer tiles along a column (4), `in1_L2L1` along a row (8):
```c++
aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [...], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : ...
aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [...], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : ...
```
Input `dma_memcpy_nd` (only 4 `in1_L3L2` queues; BD ids reused across shim tiles):
```c++
aiex.npu.dma_memcpy_nd(%arg0[...][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1, metadata = @in0_L3L2_0} : ...
aiex.npu.dma_memcpy_nd(%arg1[...][2, 16, 64, 16][64, 8192, 128, 1]) {id = 3, metadata = @in1_L3L2_0} : ...
```

## Task 3 - Writing the Output

The 4 `L1L2` FIFO queues of a column joined to one `L2L3` queue, offsets `0/256/512/768`:
```c++
aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [...], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
aie.objectfifo.link [@out_L1L2_0_0, @out_L1L2_0_1, @out_L1L2_0_2, @out_L1L2_0_3] -> [@out_L2L3_0]([0, 256, 512, 768] [])
```
Output `dma_memcpy_nd` per column (`a` in two passes, id 0 and 5), then two `dma_wait` per `L2L3` queue:
```c++
aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 4, 16, 16][64, 16, 128, 1]) {id = 0, metadata = @out_L2L3_0} : ...
// ...
aiex.npu.dma_wait {symbol = @out_L2L3_0}   // once per pass -> two waits per queue
```

## Task 4 - Testing

Run with `make run_matmul`.


## Ideas: Group Specific Component

#### CuTile
Our ideas were:
- trying to reproduce the paper https://www.mdpi.com/2079-9292/15/5/1034 (speedups of kernel fusing)
- trying to implement + benchmark some ideas of this https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf (e.g. more tiling for reuse on different hardware levels)

#### XDNA
- Reproduce the light-field tensor-ring decomposition from assignment 6 on XDNA
- 'acspx,bspy->abcyx'