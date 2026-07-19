# Submission 05: Contraction Interface and L2 Optimization

In this assignment you will build a high-level configuration interface for tensor contractions, implement an optimizer that manipulates those configurations, and use it to derive and benchmark an L2-optimized cuTile kernel.

All code should be written in `src/`.


**Use FP16 data type for tensor inputs and outputs, accumulate in FP32.**
We assume row-major order for all tensors.

---

## Task 1: Config Class


```{literalinclude} ../../assignments/05_assignment/src/config.py
:language: python
:pyobject: Config
```

---

## Task 2: Generating a Basic Config

Write a function `generate_config` that takes an einsum string and a list of shapes for the input tensors (the output shape is implied by the einsum) and returns a basic `Config`.

```{literalinclude} ../../assignments/05_assignment/src/config.py
:language: python
:pyobject: generate_config
```
---

## Task 3: Optimizer Class

Implement a class `Optimizer` that wraps a `Config` and exposes methods to transform it.

a) **Implement** the function `split_dim(dim_id: int, outer_size: int, inner_size: int)`.

```{literalinclude} ../../assignments/05_assignment/src/optimizer.py
:language: python
:pyobject: Optimizer.split_dim
```
b) **Implement** the function `fuse_dims(dim_id_a: int, dim_id_b: int)`.

```{literalinclude} ../../assignments/05_assignment/src/optimizer.py
:language: python
:pyobject: Optimizer.fuse_dims
```

c) **Implement** the function `permute_dims(permutation: list[int])`.
```{literalinclude} ../../assignments/05_assignment/src/optimizer.py
:language: python
:pyobject: Optimizer.permute_dims
```

d) **Implement** the function `make_executable()`.
```{literalinclude} ../../assignments/05_assignment/src/optimizer.py
:language: python
:pyobject: Optimizer.make_executable
```

e) **Implement** the function `verify()`.
```{literalinclude} ../../assignments/05_assignment/src/optimizer.py
:language: python
:pyobject: Optimizer.verify
```
---

## Task 4: L2-Optimized Batched Contraction

NOTE: wenn ab 5 dimensionen performance schlechter, dann siehe assume_div_by hints der optional task in week 2.

Consider the batched matrix multiplication expressed as `cmk, ckn -> cmn` with dimension sizes $|c| = 4$, $|m| = |n| = |k| = 4096$.

a) Use your `generate_config` function from Task 2 to produce the initial `Config` for this contraction. **Report** the resulting config.

Output:
```{literalinclude} src/task4a.out
```


b) Use your `Optimizer` and the implemented functions from Task 3 to transform the basic config into an L2-optimized one, following the general L2-reuse pattern from the lecture.
```
config.dim_sizes = [ [...], |m_l2|, |n_l2|, |m_prim|, |n_prim|, |k_prim|]
```

**Choose** the sizes for `m_l2`, `m_prim`, `n_l2`, `n_prim` and **justify** your choice with respect to L2 cache reuse.
**Report** the final config.

### prim sizes
First we want to choose the optimal `m_prim, n_prim, k_prim` sizes of one mma instruction. 

We want to fit as many primitive operations of a matrix multiplication into one operation of mma as possible which is `m_prim * n_prim * k_prim`. In one mma operation the maximal memory is th max shared memory per block, which is 48 KiB for our machine. Using FP32 (4 bytes) for accumulation and FP16 (2 bytes) for inputs, the required memory for one mma is `mma_size = (2 * m_prim * k_prim + 2 * k_prim * n_prim + 4 * m_prim * n_prim)` bytes. 

Therefore want to maximize `m_prim * n_prim * k_prim` s.t. `mma_size = max_shared_memory_per_block`. 

Assuming `m_prim = n_prim` due to symmetry, leaves the optimization problem:

max `m_prim^2 * k_prim` s.t. `4 * m_prim * k_prim + 4 * m_prim^2 = max_shared_memory_per_block`

This is solved for `m_prim = sqrt(max_shared_memory_per_block/12)`

In the case of 48 KiB of shared memory, `m_prim = n_prim = 64` and `k_prim = max_shared_memory_per_block / (4 * m_prim) - m_prim = 128` is optimal.

### L2 sizes
Next we want to choose `m_l2` and `n_l2` such that they fit into the L2 cache. The size of the L2 cache on our machine is 24MiB. Each kernel block loads `k//k_prim=32` tiles of A and B of size `k_prim * m_prim` and `k_prim * n_prim` respectively with `128*64*2 = 16384` bytes. So in total we can load `24 MiB / 16384 bytes = 1536` tiles of A and B into the L2 cache, resulting in `1536 / 32 = 48` kernel blocks fitting into the L2 cache. So we can choose `m_l2 = n_l2 = 24` to have 48 blocks working on different `m_l2, n_l2` tiles while reusing the same `k_outer` tiles in L2. Another option is `m_l2 = 32` and `n_l2 = 16`, also resulting in 48 blocks in L2. This might be better since both are a power of two.

c) Implement the kernel

Implement a cuTile kernel that computes `cmk, ckn -> cmn` following your optimized config from b). **Verify** correctness of your kernel.

d) Use `triton.testing.do_bench` (or a similar benchmark function provided by cuTile/Torch) to measure the average kernel runtime. **Report** the achieved performance in TFLOPS.
**Compare** the performance of your L2-optimized kernel to a baseline kernel that maps BIDs in plain row-major order over `(c, m, n)` without any splitting or permuting. **Report** your findings.

```{literalinclude} src/task4.py
:language: python
:pyobject: task_c_and_d
```

```{literalinclude} src/task4.py
:language: python
:pyobject: multiply
```

![alt text](../../assignments/05_assignment/src/task4_results.png)

The optimized kernel achieves 5.66 TFLOPS, while the baseline kernel achieves 15.14 TFLOPS. The optimized kernel is slower than the baseline kernel, which is unexpected. One possible reason for this could be that the chosen tile sizes for L2 optimization are not optimal (maybe a wrong assumption in b). Another reason could be that the overhead of managing the more complex tiling and scheduling in the optimized kernel outweighs the benefits of improved cache reuse. Further analysis and tuning of the tile sizes and scheduling strategy may be necessary to achieve better performance with the optimized kernel.