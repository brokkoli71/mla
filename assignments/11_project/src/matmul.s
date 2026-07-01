  .section .text.conv,"ax",@progbits
  .globl conv
  .p2align 4
  .type conv,@function
conv:
//nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
nopv  ; nopa  ; nopb  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cml0, [p0], #64  ; vldb x0, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cmh0, [p0], #64  ; vldb x1, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cml1, [p0], #64  ; vldb x2, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cmh1, [p0], #64  ; vldb x3, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cml2, [p0], #64  ; vldb x4, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cmh2, [p0], #64  ; vldb x5, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cml3, [p0], #64  ; vldb x6, [p1], #64  ; nops  ; nopm  ; nopx
// nopv  ; vlda.conv.fp32.bf16 cmh3, [p0], #64  ; vldb x7, [p1], #64  ; nops  ; nopm  ; nopx
mov r4, #60
movxm r3, #16256
vbcst.16 x8, r3
vmov x9, x8
nopv                                                  ; vlda.conv.fp32.bf16 cml0, [p0], #64  ; nopb  ; nops  ; mov p2, p0  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh0, [p0], #64  ; nopb  ; nops  ; mov p3, p1  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cml1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cml0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cml1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26]   ; vlda.conv.fp32.bf16 cml0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26]   ; vlda.conv.fp32.bf16 cml1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26]   ; vlda.conv.fp32.bf16 cml0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh0, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26]   ; vlda.conv.fp32.bf16 cml1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; vlda.conv.fp32.bf16 cmh1, [p0], #64  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26]   ; nopa  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; nopa  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26]   ; nopa  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; nopa  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26]   ; nopa  ; nopb  ; nops  ; nopm  ; nopx
nopv                                                  ; nopa  ; nopb  ; nops  ; nopm  ; nopx
vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26]   ; nopa  ; nopb  ; nops  ; nopm  ; nopx
vst.flush.512.conv [p2, sf, r26]
nopv                          ; nopa                                ; vldb x0, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x1, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x2, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x3, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x0, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x1, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x2, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x3, [p1], #64    ; nops                                ; nopm                    ; nopx
vmul.f dm2, y0, y4, r4        ; nopa                                ; vldb x0, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x1, [p1], #64    ; nops                                ; nopm                    ; nopx
vmul.f dm3, y1, y4, r4        ; nopa                                ; vldb x2, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x3, [p1], #64    ; nops                                ; nopm                    ; nopx
vmul.f dm2, y0, y4, r4        ; nopa                                ; vldb x0, [p1], #64    ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x1, [p1], #64    ; nops                                ; nopm                    ; nopx
vmul.f dm3, y1, y4, r4        ; nopa                                ; vldb x2, [p1], #64    ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p3, sf, r26]                                ; nopm                    ; nopx
nopv                          ; nopa                                ; vldb x3, [p1], #64    ; nops                                ; nopm                    ; nopx
vmul.f dm2, y0, y4, r4        ; nopa                                ;                       ; vst.push.576.conv.bfp16ebs8.fp32 dm3, [p3, sf, r26]                                ; nopm                    ; nopx
nopv                          ; nopa                                ;                       ; nops                                ; nopm                    ; nopx
vmul.f dm3, y1, y4, r4        ; nopa                                ;                       ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p3, sf, r26]                                ; nopm                    ; nopx
nopv                          ; nopa                                ;                       ; nops                                ; nopm                    ; nopx
vmul.f dm2, y0, y4, r4        ; nopa                                ;                       ; vst.push.576.conv.bfp16ebs8.fp32 dm3, [p3, sf, r26]                                ; nopm                    ; nopx
nopv                          ; nopa                                ;                       ; nops                                ; nopm                    ; nopx
vmul.f dm3, y1, y4, r4        ; nopa                                ;                       ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p3, sf, r26]                                ; nopm                    ; nopx
nopv                          ; nopa                                ;                       ; nops                                ; nopm                    ; nopx

nopv                          ; nopa                                ; nopb                  ; vst.push.576.conv.bfp16ebs8.fp32 dm3, [p3, sf, r26]           ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb                  ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb                  ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p3, sf, r26]           ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb                  ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb                  ; vst.push.576.conv.bfp16ebs8.fp32 dm3, [p3, sf, r26]           ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb                  ; vst.flush.512.conv [p3, sf, r26]                              ; nopm                    ; nopx

nopv  ; nopa  ; nopb  ; nops  ; nopm  ; ret lr
nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx
nopv                          ; nopa                                ; nopb               ; nops                                ; nopm                    ; nopx


.Lfunc_end0:
  .size conv, .Lfunc_end0-conv



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
 
nopv                          ; vlda.fill.512 [p0, lf0, r24]     ; vldb.fill.512 [p1, lf1, r25]   ; nops                                ; nopm                    ; nopx
nopv                          ; vlda.pop.576 ex0 [p0 lf0, r24]   ; vldb.pop.576 ex4 [p1 lf1, r25] ; nops                                ; nopm                    ; nopx
nopv                          ; vlda.pop.576 ex1 [p0 lf0, r24]   ; vldb.pop.576 ex5 [p1 lf1, r25] ; nops                                ; nopm                    ; nopx
nopv                          ; vlda.pop.576 ex2 [p0 lf0, r24]   ; vldb.pop.576 ex6 [p1 lf1, r25] ; nops                                ; nopm                    ; nopx
nopv                          ; vlda.pop.576 ex3 [p0 lf0, r24]   ; vldb.pop.576 ex7 [p1 lf1, r25] ; nops                                ; nopm                    ; nopx




.Lfunc_end1:
  .size matmul, .Lfunc_end1-matmul

