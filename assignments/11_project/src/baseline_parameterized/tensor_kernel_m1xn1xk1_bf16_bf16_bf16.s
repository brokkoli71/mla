  .globl tensor_kernel_m1xn1xk1_bf16_bf16_bf16
  .p2align 4
  .type tensor_kernel_m1xn1xk1_bf16_bf16_bf16,@function
tensor_kernel_m1xn1xk1_bf16_bf16_bf16:

// m2 n2 m1 n1 m0 n0 += m2 k1 m1 m0 k0, n2 k1 n1 k0 n0
//   += : read==write bf16

// |m2| * |n2| >= 2
// |k1| >= 6 && |k1| % 2 == 0
// |m1| = |n1| = 2
// |m0| = |n0| = |k0| = 8

// Parameters:
// p0 = IN0
// p1 = IN1
// p2 = OUT
// r0 = |m2|
// r1 = |n2|
// r2 = |k1|

// Variables:
// r3  = m1n1_i % r1
// r4  = jumpback in IN0 (|k1| * r19)
// r5  = jumpback in IN1 (|n2| * r4)

// r6 = r4 or 0 (current jump in IN0)
// r7 = r5 or 0 (current jump in IN1)

// m0 = r6
// m1 = r7

// r16 = k1_i
// r17 = m2n2_i

// r19 = -1 * 128 (8 * 8 block * 2 Byte) * 2 (|m1| or |n1|)

// Round to nearest even (fp32 -> bfp16 and bf16): mov crrnd, #12

 
//nopv                          ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx

  nopv                          ; vlda.conv.fp32.bf16 cml0, [p2], #64 ; vldb x8, [p1], #64  ; nops                                ; mov p3, p2               ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh0, [p2], #64 ; vldb x9, [p1], #64  ; nops                                ; movxm r19, #-128 * 2
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; nops                                ; nopm                     ; nopx

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; nopm                     ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml1, [p2], #64 ; vldb x9, [p1], #64  ; nops                                ; nopm                     ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; nops                                ; movxm r18, #16256
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vbcst.16 x10, r18        ; movx r20, #52
  nopv                          ; vlda.conv.fp32.bf16 cmh1, [p2], #64 ; vldb x9, [p1], #64  ; nops                                ; vbcst.16 x11, r18        ; movx r21, #53
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; nops                                ; vshuffle x6, x8, x9, r20 ; movx r22, #60

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; movx r3, #0
  nopv                          ; vlda.conv.fp32.bf16 cml2, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; mul r17, r0,  r1 
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; nops                                ; vshuffle x6, x8, x9, r20 ; mul r4,  r19, r2  
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; add r17, r17, #-2 
  nopv                          ; vlda.conv.fp32.bf16 cmh2, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; movx r24, #1
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; nops                                ; vshuffle x6, x8, x9, r20 ; mul r5, r1, r4

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; add r3,  r3,  #1 
  nopv                          ; vlda.conv.fp32.bf16 cml3, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; ltu r6,  r3,  r1 
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; mul r3,  r3,  r6 
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; xor r7,  r6,  r24
  nopv                          ; vlda.conv.fp32.bf16 cmh3, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; mov r0, #780             ; mul r6,  r6,  r4
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; mul r7,  r7,  r5

  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; add r16, r2, #-6
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_end_warm_up
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; eqz r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; mov m0, r6               ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; mov m1, r7               ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_k1_start_warm_up: // {{{
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; add r16, r16, #-2
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_start_warm_up
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nez r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_k1_end_warm_up: // }}}
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; paddb [p1], m1      ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; padda [p0], m0                      ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; vlda.conv.fp32.bf16 cml0, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cmh0, [p2], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                     ; nopx

  vmac.f dm2, dm0, ex2, ex4, r0 ; nopa                                ; vldb x8, [p1], #64  ; nops                                ; jz r17, #.l_mn_loop_end
  nopv                          ; vlda.conv.fp32.bf16 cml1, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmac.f dm2, dm2, ex3, ex4, r0 ; vlda.conv.fp32.bf16 cmh1, [p2], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; nopm                     ; nopx                       // Delay Slot 4
  nopv                          ; vlda.conv.fp32.bf16 cml2, [p2], #64 ; vldb x8, [p1], #64  ; nops                                ; nopm                     ; nopx                       // Delay Slot 3
  vmac.f dm3, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh2, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1
 
.p2align 4
.l_mn_loop_start:
  vmac.f dm3, dm3, ex3, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cml2, [p3], #64  ; vshuffle x7, x8, x9, r21 ; add r3,  r3,  #1
  nopv                          ; vlda.conv.fp32.bf16 cml3, [p2], #64 ; vldb x9, [p1], #64  ; vst.conv.bf16.fp32 cmh2, [p3], #64  ; nopm                     ; ltu r6,  r3,  r1
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vst.conv.bf16.fp32 cml2, [p3, #128] ; vshuffle x6, x8, x9, r20 ; mul r3,  r3,  r6
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cmh2, [p3, #192] ; vshuffle x7, x8, x9, r21 ; xor r7,  r6,  r24
  nopv                          ; vlda.conv.fp32.bf16 cmh3, [p2], #64 ; vldb x9, [p1], #64  ; vst.conv.bf16.fp32 cml3, [p3], #64  ; nopm                     ; mul r6,  r6,  r4
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vst.conv.bf16.fp32 cmh3, [p3], #192 ; vshuffle x6, x8, x9, r20 ; mul r7,  r7,  r5

  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cml3, [p3], #64  ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cmh3, [p3], #64  ; vshuffle x7, x8, x9, r21 ; add r16, r2, #-6
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_end_loop
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; eqz r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; mov m0, r6               ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; mov m1, r7               ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_k1_start_loop: // {{{
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; add r16, r16, #-2
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_start_loop
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nez r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4  // }}}
.l_k1_end_loop:
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; paddb [p1], m1      ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; padda [p0], m0                      ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; vlda.conv.fp32.bf16 cml0, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cmh0, [p2], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                     ; add r17, r17, #-1

  vmac.f dm2, dm0, ex2, ex4, r0 ; nopa                                ; vldb x8, [p1], #64  ; nops                                ; jnz r17, #.l_mn_loop_start
  nopv                          ; vlda.conv.fp32.bf16 cml1, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmac.f dm2, dm2, ex3, ex4, r0 ; vlda.conv.fp32.bf16 cmh1, [p2], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; nopm                     ; nopx                       // Delay Slot 4
  nopv                          ; vlda.conv.fp32.bf16 cml2, [p2], #64 ; vldb x8, [p1], #64  ; nops                                ; nopm                     ; nopx                       // Delay Slot 3
  vmac.f dm3, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh2, [p2], #64 ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_mn_loop_end:
  vmac.f dm3, dm3, ex3, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cml2, [p3], #64  ; vshuffle x7, x8, x9, r21 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml3, [p2], #64 ; vldb x9, [p1], #64  ; vst.conv.bf16.fp32 cmh2, [p3], #64  ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vst.conv.bf16.fp32 cml2, [p3, #128] ; vshuffle x6, x8, x9, r20 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cmh2, [p3, #192] ; vshuffle x7, x8, x9, r21 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh3, [p2], #64 ; vldb x9, [p1], #64  ; vst.conv.bf16.fp32 cml3, [p3], #64  ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vst.conv.bf16.fp32 cmh3, [p3], #192 ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cml3, [p3], #64  ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; vst.conv.bf16.fp32 cmh3, [p3], #64  ; vshuffle x7, x8, x9, r21 ; add r16, r2, #-6
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_end_cool_down
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; eqz r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_k1_start_cool_down: // {{{
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ;  add r16, r16, #-2
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex1, dm4       ; movxm p4, #.l_k1_start_cool_down
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nez r23, r16

  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; jnzd r23, r23, p4
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx                       // Delay Slot 5
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 4
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx                       // Delay Slot 3
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx                       // Delay Slot 2
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx                       // Delay Slot 1

.p2align 4
.l_k1_end_cool_down: // }}}
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64  ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64  ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r20 ; nopx

  vmac.f dm0, dm0, ex2, ex4, r0 ; nopa                                ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r20 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; nopa                                ; nopb                ; nops                                ; vshuffle x7, x8, x9, r21 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                     ; nopx
  vmul.f dm4, y3, y5, r22       ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                     ; nopx

  vmac.f dm0, dm0, ex0, ex4, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex4, dm4       ; nopm                     ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                     ; nopx

  vmac.f dm0, dm0, ex2, ex4, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx

  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cml0, [p3], #64  ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cmh0, [p3], #64  ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cml2, [p3, #128] ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cmh2, [p3, #192] ; nopm                     ; ret lr
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cml1, [p3], #64  ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cmh1, [p3], #192 ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cml3, [p3], #64  ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; vst.conv.bf16.fp32 cmh3, [p3], #64  ; nopm                     ; nopx
  nopv                          ; nopa                                ; nopb                ; nops                                ; nopm                     ; nopx

.Lfunc_end0:
  .size tensor_kernel_m1xn1xk1_bf16_bf16_bf16, .Lfunc_end0-tensor_kernel_m1xn1xk1_bf16_bf16_bf16
