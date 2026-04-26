# Assignment 03: Matrix Multiplication with cuTile


## Task 1: FP32 vs FP16 Performance


---

## Task 2: Simple Matrix Multiplication Kernel



---

## Task 3: Benchmarking the Matrix Multiplication Kernel



a) Benchmark your kernel with tile shapes `(64, 64, 64)` for square matrix multiplications of sizes:


b) Fix the matrix size at `2048 × 2048 × 2048`, as well as `512 × 512 × 512`, and benchmark all tile shape combinations (27 total):


**Visualize** your results as a **heatmap** with `m_tile` on one axis and `n_tile` on the other, fixing `k_tile = 64`.

**Report** the best-performing tile shape combination.

---

## Task 4: L2 Cache Optimization via Block Swizzling

a) **Your task** is to implement a swizzled matrix multiplication kernel. The requirements are the same as in _Task 2_, except block IDs should not be mapped in row-major order. Swizzle them for L2 cache reuse. You can assume a contraction dimension size of `4096`.

**Report** how you choose to map the BIDs and why. **Verify** correctness of the swizzled kernel against `torch.matmul`.

b) Repeat the tile shape sweep from _Task 3b_ for your swizzled kernel and **report** the best performing tile shape combination. **Compare** the performance of your swizzled kernel to the performance of your kernel from _Task 2_ for a matrix multiplication of shape `8192 × 8192 × 4096` (`m × n × k`).

