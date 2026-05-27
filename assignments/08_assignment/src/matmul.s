  .file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:
// Computes out += in0 * in1
// L1 tensor views:
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk
//   in1: rqkn
//   out: pqmn

// TODO: implement tensor kernel

  // load inputs in1
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  vldb x2, [p1, #128]
  vldb x3, [p1, #192]
  nop
  nop
  nop
  nop
  nop
  nop
  // load inputs in0
  # 1 q-stride = 8 for k, 8 for m, 8 for r, 2 byte for bf16 = 1024
  vlda.conv.fp32.bf16	 cml0, [p0,#0]
  vlda.conv.fp32.bf16	 cmh0, [p0,#64]
  # to padd by 1024
  padds [p0], #256
  padds [p0], #256
  padds [p0], #256
  padds [p0], #256
  vlda.conv.fp32.bf16	 cml1, [p0,#0]
  vlda.conv.fp32.bf16	 cmh1, [p0,#64]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop

  // transpose in1
  mova	r0, #52
  mova	r1, #53
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1


  nop
  nop
  nop
  nop
  nop
  // x0 to x3 are free again -> can be used for in0
  vconv.bfp16ebs8.fp32 ex0, dm0
  vconv.bfp16ebs8.fp32 ex1, dm1

  nop
  nop
  nop

  nop
  nop
  nop
  nop
  nop

  //TODO: convert the second half also and save it to ex3
  // BF16 -> fp32 of in1
  movxm r0, #16256
  vbcst.16 x0, r0
  nop
  vmov x1, x0
  mova r0, #60
  vmul.f dm2, y1, y0, r0
  nop
  nop
  nop
  nop
  nop
  // fp32 -> bfp16
  vconv.bfp16ebs8.fp32 ex2, dm2
  nop
  nop
  nop

  // all inputs are in ex, outputs can be loaded to dm
  //load outputs
  vlda.conv.fp32.bf16	 cml0, [p2], #64
  vlda.conv.fp32.bf16	 cmh0, [p2], #64
  vlda.conv.fp32.bf16	 cml1, [p2], #64
  vlda.conv.fp32.bf16	 cmh1, [p2], #64
  vlda.conv.fp32.bf16	 cml2, [p2], #64
  vlda.conv.fp32.bf16	 cmh2, [p2], #64
  vlda.conv.fp32.bf16	 cml3, [p2], #64
  vlda.conv.fp32.bf16	 cmh3, [p2], #64
  
  nop
  nop
  nop
  // matrix multiplication
  mova r0, #780
  vmac.f dm0, dm0, ex0, ex2, r0
  vmac.f dm1, dm1, ex1, ex2, r0 //TODO: maybe swap output of dm1 and dm2
  vmac.f dm2, dm2, ex0, ex3, r0
  vmac.f dm3, dm3, ex1, ex3, r0

  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  // todo: store the rest of the outputs
  vst.conv.bf16.fp32 cml0, [p2, #0]
  vst.conv.bf16.fp32 cmh0, [p2, #64]
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
