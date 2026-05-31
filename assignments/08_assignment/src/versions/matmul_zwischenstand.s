  .file "matmul.s"
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
//   in1: rqkn
//   out: pqmn
  // BF16 -> fp32
  // vbcst.16 x0, r2
  // nop
  // nop
  // vmov x1, x0
  // vmul.f dm1, y1, y0, r3


// TODO: implement tensor kernel
  //load outputs
  vlda.conv.fp32.bf16	 cml0, [p0, #0]; vldb x0, [p1, #0]
  vlda.conv.fp32.bf16	 cmh0, [p0, #64]; vldb x1, [p1, #64]
  #movxm r2, #16256
  #mov r3, #60
  vlda.conv.fp32.bf16	 cml2, [p2, #0]
  vlda.conv.fp32.bf16	 cmh2, [p2, #64]
  nop
  nop
  nop
  nop
  mov	r0, #52
  mov	r1, #53
  // transpose
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex8, dm0
  vshuffle x3, x0, x1, r1
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  nop
  // fp32 -> bfp16
  vconv.bfp16ebs8.fp32 ex0, dm1
  nop
  nop
  mova r4, #780
  // matrix multiplication
  vmac.f dm2, dm2, ex8, ex0, r4

  // 2nd
  mov p3, p0 //in0
  mov p4, p1 //in1
  mov p5, p2 //out
  padds [p3], #128 //in0 zum nächsten k schieben
  padds [p4], #256 //in1 zum nächsten k schieben


  vlda.conv.fp32.bf16	 cml0, [p3, #0]; vldb x0, [p4, #0]
  vlda.conv.fp32.bf16	 cmh0, [p3, #64]; vldb x1, [p4, #64]
  #movxm r2, #16256
  #mov r3, #60
  nop
  nop
  nop
  nop
  nop
  nop
  mov	r0, #52
  mov	r1, #53
  // transpose
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex8, dm0
  vshuffle x3, x0, x1, r1
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // fp32 -> bfp16
  nop
  vconv.bfp16ebs8.fp32 ex0, dm1
  nop
  nop
  mova r4, #780
  // matrix multiplication
  vmac.f dm2, dm2, ex8, ex0, r4

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
