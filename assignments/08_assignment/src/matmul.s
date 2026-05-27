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
  //load outputs
  vlda.conv.fp32.bf16	 cml2, [p2, #0]
  vlda.conv.fp32.bf16	 cmh2, [p2, #64]
  nop
  nop
  nop
  vldb x0, [p1, #0]
  vldb x1, [p1, #64]
  nop
  nop
  nop
  nop
  nop
  nop

  vlda.conv.fp32.bf16	 cml0, [p0, #0]
  vlda.conv.fp32.bf16	 cmh0, [p0, #64]
  nop
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex8, dm0
  nop
  nop
  nop

  // transpose
  mova	r0, #52
  mova	r1, #53
  vshuffle x2, x0, x1, r0
  vshuffle x3, x0, x1, r1

  nop
  nop
  nop

  // BF16 -> fp32
  movxm r0, #16256
  vbcst.16 x0, r0
  nop
  vmov x1, x0
  mova r0, #60
  vmul.f dm1, y1, y0, r0
  nop
  nop
  nop
  nop
  nop
  // fp32 -> bfp16
  vconv.bfp16ebs8.fp32 ex0, dm1
  nop
  nop
  nop


  // matrix multiplication
  mova r0, #780
  vmac.f dm2, dm2, ex8, ex0, r0

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
  vst.conv.bf16.fp32 cml2, [p2, #0]
  vst.conv.bf16.fp32 cmh2, [p2, #64]
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
