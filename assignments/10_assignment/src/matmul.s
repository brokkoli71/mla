  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:

// Performance:
//   ints = 3 + 3*6 + 2*2*6 + 4*6 + 9  = 78
//   macs = 4 * 8 * (8 * 8 * 8)  = 16384
//   macs/cycle = 16384 / 78 = 210.05
//   GFLOPS = 210.05 * 2 * 1.8 = 756.18
//   %peak = 756.18 / (1.8 * 1024) = 0.41

// L1 layouts (BF16):
//   in0: prmk = 2x8x8x8 BF16
//   in1: rqkn = 8x2x8x8 BF16
//   out: pqmn = 2x2x8x8 BF16
 
//nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx

  nopv                          ; vlda.conv.fp32.bf16 cml0, [p2], #64 ; vldb x8, [p1], #64 ; nops                                ; mov p3, p2              ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh0, [p2], #64 ; vldb x9, [p1], #64 ; nops                                ; mov p4, p0              ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; nops                                ; mov crrnd, #12          ; nopx

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; movxm m0, #1024
  nopv                          ; vlda.conv.fp32.bf16 cml1, [p2], #64 ; vldb x9, [p1], #64 ; padds [p4], m0                      ; movxm r1, #16256
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; nops                                ; vbcst.16 x10, r1        ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; vldb x8, [p1], #64 ; nops                                ; vbcst.16 x11, r1        ; movx r6, #60
  nopv                          ; vlda.conv.fp32.bf16 cmh1, [p2], #64 ; vldb x9, [p1], #64 ; nops                                ; mov r0, #780            ; movx r2, #52
  nopv                          ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; nops                                ; vshuffle x6, x8, x9, r2 ; movx r3, #53

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml2, [p2], #64 ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; nops                                ; vshuffle x6, x8, x9, r2 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh2, [p2], #64 ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; nops                                ; vshuffle x6, x8, x9, r2 ; nopx

  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cml3, [p2], #64 ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  nopv                          ; vlda.conv.fp32.bf16 cmh3, [p2], #64 ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r2 ; nopx

.rept 2
// k mod 2 == 0
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r2 ; nopx

// k mod 2 == 1
  vmac.f dm0, dm0, ex2, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p0], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
.endr
// k == 4
  vmac.f dm0, dm0, ex0, ex4, r0 ; vlda.conv.fp32.bf16 cmh4, [p0], #64 ; vldb x8, [p1], #64 ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; vldb x9, [p1], #64 ; vconv.bfp16ebs8.fp32 ex0, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; vlda.conv.fp32.bf16 cml4, [p4], #64 ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; vlda.conv.fp32.bf16 cmh4, [p4], #64 ; nopb               ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex1, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; vshuffle x6, x8, x9, r2 ; nopx

// k == 5
  vmac.f dm0, dm0, ex2, ex4, r0 ; nopa                                ; nopb               ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex2, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; vshuffle x6, x8, x9, r2 ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; nopa                                ; nopb               ; nops                                ; vshuffle x7, x8, x9, r3 ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex3, dm4       ; nopm                    ; nopx
  vmul.f dm4, y3, y5, r6        ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                    ; nopx

// k == 6
  vmac.f dm0, dm0, ex0, ex4, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  vmac.f dm2, dm2, ex1, ex4, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex4, dm4       ; nopm                    ; nopx
  vmac.f dm1, dm1, ex0, ex5, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  vmac.f dm3, dm3, ex1, ex5, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vconv.bfp16ebs8.fp32 ex5, dm4       ; nopm                    ; nopx

// k == 7
  vmac.f dm0, dm0, ex2, ex4, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  vmac.f dm2, dm2, ex3, ex4, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  vmac.f dm1, dm1, ex2, ex5, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  vmac.f dm3, dm3, ex3, ex5, r0 ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx

  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cml0, [p3], #64  ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cmh0, [p3], #64  ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cml2, [p3, #128] ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cmh2, [p3, #192] ; nopm                    ; ret lr
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cml1, [p3], #64  ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cmh1, [p3], #192 ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cml3, [p3], #64  ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; vst.conv.bf16.fp32 cmh3, [p3], #64  ; nopm                    ; nopx
  nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx

.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul

