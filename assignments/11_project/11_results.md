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
As we found no faster way of converting BF16 to BFP16 on the fly without occupying the vector unit, our approach was to convert all the values beforehand, keeping them in the L1 memory of the compute tile and afterwards using this values for multiplication.

//TODO: On the higher level this makes sense 
- dadurch dass wir nicht alles in register lassen können, müssen wir mehrfach laden.
- könnte schneller sein bei großen matrizen
On the higher level this makes sense. 
The matrix multiplication is implemented by iteratively computing $M=N=16$ blocks of the result matrix. 4 `vmac.f` instructions (see code above) compute the results for 8 steps along the $K$ dimension, which will be accumulated into 4 dm registers. This block size is capped by the maximum amount of values kept in the dm accumulator registers (theoretically there is one spare register but that is used for ...).
For higher $K$ values we can not keep input values in the registers when they are used for a second 16x16 result block. Therefore, we need to reload them. For the on-the-fly-conversion implementation that means converting this values again. Hiding this conversion is not fully paralelisable because of the required `vmul.f` operations.
// aber overhead durch previous conversion (nur in der größe des inputs, nicht in der größe der benötigten operationen)


#### expected speedup
- O notation wie auf folien
- overhead konkret ausrechnen schnittpunkt finden


#### Size limitations
As there is no conceptional speedup expected with our implementation when varying the contraction dimension $K$, we keep it fixed to 64.
This work is constrained to matrix multiplications on one compute tile. Because we want to keep the whole converted matrices in the L1 memory, we are limited on the size of the input matrices. Also, we will only implement multiplications with side length $M=N$ being a power of two.
For $K=64$ the maximum size of $M,N$ is ...
...
The smallest matrix size we implemented is $M=N=16$. As the $M=N=K=16$ block can be fully kept within the registers, the previous on-the-fly-conversion kernel would not execute duplicate conversions and there is no benefit of preconverting the values. Therefore, smaller kernel sizes would not make sense to implement here.


- aufgrund von double buffering und instruktionscache (?) können wir nur 31KiB von 64KiB des L1 Cache nutzen
31*1024 byte / 2byte per BF16 value / 2 matrizen / 64 für K = 124 < 128, deshalb passen N=M=128 nicht mehr. außerdem bräuchte es noch für den output speicher


#### Implementierung

#### Auswertung
