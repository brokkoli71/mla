## Group Specific Component - Un-fusing the MatMul kernel
In this work, we try to optimize the throughput of multiplication of two matrices in BF16 data format on an AMD XDNA 2 NPU using the AIE-API. 

In the following we use the following naming of matrices and dimensions: $A\times B = C, A\in \mathbb R^{M\times K}, B\in \mathbb R^{K\times N}$

The AIE-API provides `vmac.f` that is able to multiply two 8x8 matrices with one instruction and a latency of ... cycles. With pipelining, the throughput can be maximized to issuing one `vmac.f` instruction per cycle. One limiting factor is that the input is expected to be in the BFP16 data format, into which the data has to be converted first.
The required format BFP16 (here, more precisely BFP16ebs8) is aiming to compress floats without losing precision for consecutive numbers in a similar order of magnitude.
It deduplicates information by grouping 8 consecutive floating point values into a representation with a shared exponent. Thus, for 8 values we have one 8-bit exponent, and for each value a sign bit and 7-bit mantissa.  
In previous exercises, we converted the data on the fly. Most of the work can be parallelized over the six functional-unit slots provided by the XDNA2 VLIW instruction word.
However, the best kernel found still needs one `vmul.f` instruction for conversion every two `vmac.f` instructions at its core. 
e.g.: 

```
  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; mov m0, r6               ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; mov m1, r7               ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
```

Optimally, we aim to schedule one `vmac.f` per cycle, resulting in the maximum throughput of matrix multiplications, as the vector unit is fully occupied calculating `vmac.f` multiplications.  
In the lecture we found no faster way of converting BF16 to BFP16 on the fly without occupying the vector unit, resulting in a slowdown of factor 1.5 compared to the optimum.
The approach we follow here was to convert all the values beforehand, keeping them in the L1 memory of the compute tile and afterwards using these values for multiplication.

On the higher level this makes sense. 
The matrix multiplication is implemented by iteratively computing $16\times16$ blocks of the result matrix. 
4 `vmac.f` instructions (see code above) compute the results for 8 steps along the $K$ dimension, which will be accumulated into 4 dm registers. 
This block size is capped by the maximum amount of values kept in the dm accumulator registers.
For higher $K$ values, we cannot keep input values in the registers when we need them for the next 16x16 result block. 
Therefore, we must reload them. 
For the on-the-fly-conversion implementation, that means converting these values again. 
This conversion latency cannot be fully hidden because of the required `vmul.f` operations.
When the values are already converted, this work would not be needed anymore.
But on the other hand we would need to convert the values beforehand. 
The hope is that at some size of $M$ and $N$ this overhead will be compensated for by the elimination of duplicate work and lead to a performance improvement.

#### Expected speedup
As mentioned earlier, the core of both versions (on-the-fly-conversion and preconverted) consists of densely packed `vmac.f` instructions that are executing the heavy lifting multiplications. 
Thus, the number of `vmac.f` is the same for both implementations and scales with the number of primitive multiplications needed for a matrix multiplication. 
In total we will need to compute $N\cdot M$ output values, accumulated over a depth of $K$, resulting in $N\cdot M \cdot K$ primitive multiplications.
Each `vmac.f` executes $8\cdot 8 \cdot 8 = 512$ primitive multiplications, so we land at $\frac {N\cdot M \cdot K}{512}$ `vmac.f` instructions.

The total expected cycles of the on-the-fly-conversion version are the number of vmacs multiplied by 1.5 (see `vmul.f` above) plus some constant overhead for the warmup (loading the first values) and cooldown (storing the last results), which cannot be effectively hidden during `vmac.f`s:

$$1.5\cdot \frac {N\cdot M \cdot K}{512} + \Theta(1)$$

In contrast, at the core of the kernel we aim to reach the optimum of one `vmac.f` per cycle plus warmup and cooldown. 
In addition, we need to convert both input matrices of size $M\times K$ and $K\times N$ beforehand.
 As we will discuss later, our implementation of the conversion requires 12 cycles for storing an 8x64 matrix block. Assuming that the input matrices are multiples of these blocks, we would need $\frac{12}{8\cdot64}(MK+KN) = \frac{3}{128}(M+N)K$ cycles plus a constant warmup for loading the first values.

$$\frac {N\cdot M \cdot K}{512} + \frac{3}{128}(M+N)K + \Theta(1)$$

Thus, ignoring constant warmup/cooldown, we expect our implementation to be faster iff:

$$\begin{aligned}
1.5\cdot \frac {N\cdot M \cdot K}{512} &> \frac {N\cdot M \cdot K}{512} + \frac{3}{128}(M+N)K \\
1.5 &> 1 + \frac{\frac{3}{128}(M+N)K}{\frac {N\cdot M \cdot K}{512}} \\
1.5 &> 1 + \frac{12 (M+N)}{N\cdot M} \\
0.5 &> \frac{12 (M+N)}{N\cdot M} \\
1 &> \frac{24 (M+N)}{N\cdot M} \\
N\cdot M &> 24 (M+N)
\end{aligned}$$

For the special case of $N=M$, we expect our implementation to be faster iff $N^2 > 48N$, i.e. $N > 48$

```{figure} figures/speedup_crossover.png
:alt: Expected cycles vs. matrix size for on-the-fly vs. pre-converted conversion
:width: 100%

Expected cycles for both implementations over the matrix size $N$ (with $M=N$, $K=64$). The curves cross at $N=48$ independently of $K$; for larger matrices pre-converting wins.
```

#### Size limitations in this work
As there is no conceptual speedup expected with our implementation when varying the contraction dimension $K$ (see above), we keep it fixed at 64.

This work is constrained to matrix multiplications on one compute tile. Because we want to keep the whole converted matrices in the L1 memory, we are limited in the size of the input matrices. Also, we will only implement multiplications with side length $M=N$ being a power of two.

For $K=64$ the maximum size of $M,N$ is derived from the usable size of the L1 cache. 
Due to double buffering and the need for instruction cache, we can only use 31KiB of the total 64KiB. For loading the 2 input matrices we need 2 bytes per BF16 value with $64N=64M$ values per matrix; therefore, the maximum $N$ is 

$$N \le {31\cdot 1024 \over 2\cdot 2 \cdot 64} = 124$$

Furthermore, the output matrix also needs to fit into the L1 cache.

The smallest matrix size we implemented is $M=N=16$. When iterating over $K$ we can keep accumulation results of the full matrix in 4 of the 5 `dm` accumulator registers. Therefore, there is no need to reload the data later, and thus the previous on-the-fly-conversion kernel would not execute duplicate conversions and there is no benefit in preconverting the values. Therefore, smaller kernel sizes would not make sense to implement here.

There are 3 remaining sizes of matrix multiplication fulfilling these restrictions, which we implemented:
$K=64, N=M\in\{16, 32, 64\}$ 

#### Implementation
The implementation is split into two kernels: a conversion kernel that rewrites both input matrices from BF16 into the BFP16 block format, and the matmul kernel that consumes the converted data. Both kernels operate entirely on the L1 memory of the compute tile, so no data movement beyond the tile is required between the two phases.

#### Conversion Kernel
The conversion kernel reads both input tensors in parallel, using the `a` and the `b` load engine at the same time.

The hardware can only produce BFP16 from FP32, not directly from BF16, so every input value has to be widened to FP32 before it can be converted. The `a` load engine does this for free: `vlda.conv.fp32.bf16` performs the widening as part of the load instruction. The `b` load engine has no such conversion variant, so for the second matrix we have to widen explicitly with a `vmul.f` against a broadcast constant of $1.0$, which costs one vector slot per loaded block.

Storing the converted data is done with the instruction pair

```
vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26]
vst.flush.512.conv               [p2, sf, r26]
```

A single store can only write 64 bytes, but one BFP16ebs8 group of 64 values occupies 72 bytes (8 blocks of 8 mantissas plus their shared exponent byte). One group therefore cannot be emitted by a single store instruction. The `vst.push` instruction resolves this by writing 64 bytes and recording the 8-byte remainder in `r26`, which it decrements by 8 on every push. `vst.flush` then reads `r26` and writes out the accumulated remainder. Since a flush can also write a maximum of 64 bytes, one flush is needed after at most eight pushes.

Both instructions are fixed to the registers `p2`, `sf` and `r26`. The pointer, the shift/format register and the remainder counter cannot be substituted by other registers. `p2` is a live store cursor that both the pushes and the flush advance on their own, so the base address of a group has to be captured with `movs` right after the flush and restored with `mov` before the next group.

We arrived at this kernel in three optimization steps, which are preserved in the directory `size16` as the files `matmul_unoptimiert.s`, `matmul.s` and `matmul_optimiert.s`.

`matmul_unoptimiert.s` is the naive implementation: it converts the two input matrices one after the other and issues a push only every second cycle, which amounts to 16 cycles per stored 8x64 matrix block.

`matmul.s` interleaves the conversion of both input matrices and uses the vector registers as buffers, so that a conversion can be issued in almost every cycle. Since the widening of the `b` matrix occupies the vector slot, the two matrices are processed in alternating groups of four pushes, each terminated by a flush and a bundle that restores the group base pointer. This brings the cost down to 12 cycles per stored 8x64 matrix block.

`matmul_optimiert.s` pushes the same idea further and holds more values in registers, so that a full group of eight pushes can be issued back to back before a flush is needed. One flush and one pointer bundle then amortize over eight pushes instead of four, which reduces the cost to 10 cycles per stored 8x64 matrix block.

Lines 63-118 of `matmul_optimiert.s` contain a further, unsuccessful optimization attempt. There, the bundle that only restores the group base pointer after each flush is removed and the `mov p2, p3` is folded into the flush bundle itself, which would save one more cycle per group. The resulting kernel produces wrong data, and we could not determine why: neither the pointer arithmetic nor the fact that the pointer move shares its bundle with the flush explains the failure, since both could be excluded individually in separate experiments.


```{literalinclude} ../../assignments/11_project/src/size16/matmul_optimiert.s
:language: assembly
:lines:5-59
```

#### Matmul Kernel
The matmul kernel reads the converted matrices back with both load engines in parallel. The load side mirrors the store side of the conversion kernel: `vlda.fill.512` and `vldb.fill.512` prefetch a 64-byte chunk, and `vlda.pop.576` and `vldb.pop.576` then hand out the 72-byte BFP16 groups, with `r24` and `r25` tracking the remainder in the same way `r26` does for the stores. A `pop` instructions needs 8 cycles.

For the output tile size of 16x16 kernel issues one `vmac.f` per cycle, which is the maximum throughput. The `vmac.f` computation results are converted back to BF16 and written out with `vst.conv.bf16.fp32`.

```{literalinclude} ../../assignments/11_project/src/size16/matmul_optimiert.s
:language: assembly
:lines:204-258
```

For the output tile size 32x32, four 16x16 Output blocks needs to be computed. Since there are only 5 accumulator registers, we need to store the computed output tiles for each block and load the new block. Loads and stores needs two intructions to handle one accumulator regiter, whereas the `vmac.f` instruction computes and read the whole accumulator register in one instruction. This forces a gap of two nops in the `vmac.f` instructions, because it has to wait until the registers are stored, before the new output tile can be loaded. Since you have 5 accumultor Register and need only 4 for the 16x16 block, one can register can be preloaded without influencing the current output block.

Attemps were made to optimise this further in `size32/matmul_optimiert.s` but without sucess.
#### Evaluation
We evaluate the efficiency of our implementation in two ways: firstly by the number of lines/cycles of the kernel code and secondly via empirical benchmarks.

###### Cycles
- zeilen zählen, warmup, cooldown. 
- ist die anzahl gleichbleibend über verschiedene größen
- entweder für größen separat oder nochmal in die formel am anfang einsetzen

###### Benchmarks

- grafik mit gemessenen werten und cycles in gestrichelt (beide y achsen über summe der daten normalisiert (mean + std))