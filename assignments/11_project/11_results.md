## Group Specific Component - Un-fusing the MatMul kernel
In this work, we try to optimize the throughput of multiplication of two matrices in BF16 data format on an AMD XDNA 2 NPU using the AIE-API. 

In the following we use the following naming of matrices and dimensions: $A\times B = C, A\in \mathbb R^{M\times K}, B\in \mathbb R^{K\times N}$

The AIE-API provides `vmac.f` that is able to multiply a two 8x8 matrices with one instruction and a latency of ... cycles. With pipelining, the thoughput can by maximized to issueing one `vmac.f` instruction per cycle. One limiting factor is, that the input is expected to be in the BFP16 data format in which the data has to be converted first.
The required format BFP16 (here, more precisely BFP16ebs8) is aiming to compress floats without loosing precision for consecutive numbers in a similar order of magnitude.
It deduplicates information by grouping 8 consecutive floating point values into a representation with a shared exponent. Thus, we have for 8 values one 8 bit exponent and each a sign bit and 7 bit mantissa.  
In previous exercises, we converted the data on the fly. most of the work can be parallelized over the six functional-unit slots provided by the XDNA2 VLIW instruction word.
Though the best kernel found, still needs one `vmul.f` instruction for conversion every two `vmac.f` instruction at its core. 
e.g.: 

```
  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; mov m0, r6               ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; mov m1, r7               ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
```

Optimally we aim to schedule one `vmac.f` per cycle resulting in the maximum throughput of matrix multiplications, as the vector unit is fully occupied calculating `vmac.f` multiplications.  
In the lecture we found no faster way of converting BF16 to BFP16 on the fly without occupying the vector unit, resulting in a slowdown of factor 1.5 compared to the optimum.
The approach we follow here, was to convert all the values beforehand, keeping them in the L1 memory of the compute tile and afterwards using this values for multiplication.

On the higher level this makes sense. 
The matrix multiplication is implemented by iteratively computing $M=N=16$ blocks of the result matrix. 
4 `vmac.f` instructions (see code above) compute the results for 8 steps along the $K$ dimension, which will be accumulated into 4 dm registers. 
This block size is capped by the maximum amount of values kept in the dm accumulator registers (theoretically there is one spare register but that is used for ...).
For higher $K$ values we can not keep input values in the registers when we need them for the next 16x16 result block. 
Therefore, we must reload them. 
For the on-the-fly-conversion implementation that means converting this values again. 
Hiding this conversion latency is not fully working because of the required `vmul.f` operations.
When the values are already converted, this work would not be needed anymore.
But on the other hand we would need to convert the values beforehand. 
The hope is that at some size of $M$ and $N$ this overhead will be compensated for by the elimination of duplicate work and lead to a performance improve.

#### Expected speedup
As mentioned earlier, the core of both versions (on-the-fly-conversion and preconverted) consists of densly packed `vmac.f` instructions that are executing the heavy lifting multiplications. 
Thus, the number of `vmac.f` is the same for both implementations and scales with the number of primitive multiplications needed for a matrix multiplication. 
In total we will need to compute $N\cdot M$ output values, accumulated over a depth of $K$, resulting in $N\cdot M \cdot K$ primitive multiplications.
Each `vmac.f` executes $8\cdot 8 \cdot 8 = 512$ primitive multiplications, so we land at $\frac {N\cdot M \cdot K}{512}$ `vmac.f` instructions.

The total expected cycles of the on-the-fly-conversion version the number of vmacs multiplied by 1.5 (see `vmul.f` above) plus some constant overhead for the warmup (loading the first values) and cooldown (storing the last results) which can not be effectively hidden during `vmac.f`s:

$$1.5\cdot \frac {N\cdot M \cdot K}{512} + \Theta(1)$$

In contrast, at the core of the kernel we aim to reach the optimum of one `vmac.f` per cycle plus warmup and cooldown. 
On the other hand we need to convert both input matrices of size $M\times K$ and $K\times N$ beforehand.
As we will discuss later, our implementation of the conversion requires 12 cycles per storing a 8x64 matrix block. Assuming that the input matrices are multiples of this blocks, we would need $\frac{12}{8\cdot64}(MK+KN) = \frac{3}{128}(M+N)K$ cycles plus a constant warmup for loading the first values.

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

For the special case of $N=M$ we expect our implementation to be faster iff $N^2 > 48N$ respectively $N > 48$

```{figure} figures/speedup_crossover.png
:alt: Expected cycles vs. matrix size for on-the-fly vs. pre-converted conversion
:width: 100%

Expected cycles for both implementations over the matrix size $N$ (with $M=N$). The curves cross at $N=48$; for larger matrices pre-converting wins.
```

#### Size limitations in this work
As there is no conceptional speedup expected with our implementation when varying the contraction dimension $K$ (see above), we keep it fixed to 64.
This work is constrained to matrix multiplications on one compute tile. Because we want to keep the whole converted matrices in the L1 memory, we are limited on the size of the input matrices. Also, we will only implement multiplications with side length $M=N$ being a power of two.

For $K=64$ the maximum size of $M,N$ is derived from the usable size of the L1 cache. 
Due to double buffering and need for instruction cache, we can only use 31KiB of the total 64KiB. For loading the 2 input matrices we need 2 bytes per BF16 value with $64N=64M$ values per matrix, therefore the maximum $N$ is 

$$N \le {31\cdot 1024 \over 2\cdot 2 \cdot 64} = 124$$

Furthermore, the output matrix also needs to fit into the L1 cache.

The smallest matrix size we implemented is $M=N=16$. As the $M=N=K=16$ block can be fully kept within the registers, the previous on-the-fly-conversion kernel would not execute duplicate conversions and there is no benefit of preconverting the values. Therefore, smaller kernel sizes would not make sense to implement here.

There are 3 remaining sizes of matrix multiplication fulfilling this restrictions, which we implemented:
$K=64, N=M\in\{16, 32, 64\}$ 

#### Implementation

#### Evaluation
We evaluate the efficiency of our implementation in two ways: firstly on the number of lines/cycles of the kernel code and secondly via empirical benchmarks.

###### Cycles
- zeilen zählen, warmup, cooldown. 
- ist die anzahl gleichbleibend über verschiedene größen
- entweder für größen separat oder nochmal in die formel am anfang einsetzen

###### Benchmarks

- grafik mit gemessenen werten und cycles in gestrichelt (beide y achsen über summe der daten normalisiert (mean + std))