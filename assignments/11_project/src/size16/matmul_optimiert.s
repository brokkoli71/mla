  .section .text.conv,"ax",@progbits
  .globl conv
  .p2align 4
  .type conv,@function
conv:
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x0, [p1], #64 ; movs p3, p1                                         ; mov p2, p0      ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x1, [p1], #64 ; movs p4, p2                                         ; mov m3, #288  ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x2, [p1], #64 ; nops                                                ; mov crrnd, #12  ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x3, [p1], #64 ; nops                                                ; movxm r3, #16256
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x4, [p1], #64 ; nops                                                ; vbcst.16 x8, r3 ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x5, [p1], #64 ; nops                                                ; vbcst.16 x9, r3 ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x6, [p1], #64 ; nops                                                ; mov r4, #60     ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x7, [p1], #64 ; nops                                                ; mov r26, #0     ; nopx
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; nops                                                ; nopm       ; nopx
nopv                   ; nopa                                ; vldb x0, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y0, y4, r4 ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x1, [p1], #64 ; nops                                                ; nopm       ; nopx
vmul.f dm4, y1, y4, r4 ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x2, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y2, y4, r4 ; vlda.conv.fp32.bf16 cml1, [p0], #64 ; vldb x3, [p1], #64 ; nops                                                ; nopm       ; nopx
vmul.f dm4, y3, y4, r4 ; vlda.conv.fp32.bf16 cmh1, [p0], #64 ; vldb x4, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx // r26 max -64 danach kaputt
nopv                   ; vlda.conv.fp32.bf16 cml2, [p0], #64 ; vldb x5, [p1], #64 ; vst.flush.512.conv [p2, sf, r26]                    ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh2, [p0], #64 ; vldb x6, [p1], #64 ; nops                                                ; mov p2, p3 ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x7, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p4], m3                      ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p3], m3                      ; vldb x0, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y0, y4, r4 ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x1, [p1], #64 ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p4 ; nopx
vmul.f dm4, y1, y4, r4 ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x2, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y2, y4, r4 ; vlda.conv.fp32.bf16 cml1, [p0], #64 ; vldb x3, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y3, y4, r4 ; vlda.conv.fp32.bf16 cmh1, [p0], #64 ; vldb x4, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml2, [p0], #64 ; vldb x5, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh2, [p0], #64 ; vldb x6, [p1], #64 ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p3 ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x7, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p4], m3                      ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p3], m3                      ; vldb x0, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y0, y4, r4 ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x1, [p1], #64 ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p4 ; nopx
vmul.f dm4, y1, y4, r4 ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; vldb x2, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y2, y4, r4 ; vlda.conv.fp32.bf16 cml1, [p0], #64 ; vldb x3, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y3, y4, r4 ; vlda.conv.fp32.bf16 cmh1, [p0], #64 ; vldb x4, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml2, [p0], #64 ; vldb x5, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh2, [p0], #64 ; vldb x6, [p1], #64 ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p3 ; nopx
nopv                   ; vlda.conv.fp32.bf16 cml0, [p0], #64 ; vldb x7, [p1], #64 ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; vlda.conv.fp32.bf16 cmh0, [p0], #64 ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p4], m3                      ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; padda [p3], m3                      ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y0, y4, r4 ; nopa                                ; nopb               ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p4 ; nopx
vmul.f dm4, y1, y4, r4 ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y2, y4, r4 ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm1, [p2, sf, r26] ; nopm       ; nopx
vmul.f dm4, y3, y4, r4 ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm2, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm0, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; vst.flush.512.conv [p2, sf, r26]                    ; mov p2, p3 ; nopx
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; ret lr
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; vst.push.576.conv.bfp16ebs8.fp32 dm4, [p2, sf, r26] ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; vst.flush.512.conv [p2, sf, r26]                    ; nopm       ; nopx
nopv                   ; nopa                                ; nopb               ; nops                                                ; nopm       ; nopx

.Lfunc_end0:
  .size conv, .Lfunc_end0-conv

  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:

// Computes out += in0 * in1
// p2 += p0 * p1  
// L1 tensor views:
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk
//   in1: qrnk
//   out: pqmn
// bfp16

// L1 layouts (BF16):
//   in0: mk = 16x64 BFP16
//   in1: nk = 16x64 BFP16
//   out: mn = 16x16 BFP16
//nopv                          ; nopa                             ; nopb                           ; nops                                ; nopm                    ; nopx
vlda.conv.fp32.bf16 cml0, [p2], #64; mov p3, p2 
vlda.conv.fp32.bf16 cmh0, [p2], #64 
vlda.conv.fp32.bf16 cml1, [p2], #64
vlda.conv.fp32.bf16 cmh1, [p2], #64
nopv                          ; vlda.fill.512 [p0, lf0, r24]        ; vldb.fill.512 [p1, lf1, r25]     ; nops                               ; mov r0, #780 ; movx r25, #0
nopv                          ; vlda.pop.576 ex0 ,[p0, lf0, r24]    ; vldb.pop.576 ex1 ,[p1, lf1, r25] ; nops                               ; mov crrnd, #12 ; movx r24, #0
nopv                          ; vlda.pop.576 ex2 ,[p0, lf0, r24]    ; vldb.pop.576 ex3 ,[p1, lf1, r25] ; nops                               ; nopm           ; nopx
nopv                          ; vlda.conv.fp32.bf16 cml2, [p2], #64 ;nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; vlda.conv.fp32.bf16 cmh2, [p2], #64 ;nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; vlda.pop.576 ex0 ,[p0, lf0, r24]    ;vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
nopv                          ; vlda.pop.576 ex2 ,[p0, lf0, r24]    ;vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
nopv                          ; vlda.conv.fp32.bf16 cml3, [p2], #64 ;nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; vlda.conv.fp32.bf16 cmh3, [p2], #64 ;nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; vlda.fill.512 [p0, lf0, r24]        ; vldb.fill.512 [p1, lf1, r25]     ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; vlda.pop.576 ex0, [p0, lf0, r24]    ; vldb.pop.576 ex1, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; vlda.pop.576 ex2, [p0, lf0, r24]    ; vldb.pop.576 ex3, [p1, lf1, r25] ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm0, dm0, ex0, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm1, dm1, ex0, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm2, dm2, ex2, ex1, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
vmac.f dm3, dm3, ex2, ex3, r0 ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cml0, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cmh0, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cml1, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cmh1, [p3], #64 ; nopm           ; ret lr
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cml2, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cmh2, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cml3, [p3], #64 ; nopm           ; nopx
nopv                          ; nopa                                ; nopb                             ; vst.conv.bf16.fp32 cmh3, [p3], #64 ; nopm           ; nopx
  
nopv                          ; nopa                                ; nopb                             ; nops                               ; nopm           ; nopx



.Lfunc_end1:
  .size matmul, .Lfunc_end1-matmul

