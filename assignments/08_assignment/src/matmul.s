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
  //nop
  //nop
  //nop
  //nop
  //nop
  //nop


// TODO: implement tensor kernel
  mov p3, p0
  mov p5, p2
  //load 8x8 in0 ;                    load 8x8 in1; copy base pointer
  vlda.conv.fp32.bf16	 cml0, [p0, #0]; vldb x0, [p1, #0]; mov p4, p1; padds [p3], #128 
  vlda.conv.fp32.bf16	 cmh0, [p0, #64]; vldb x1, [p1, #64]; padds [p4], #256
    //load 2nd 8x8 in0 ;                    load 2nd 8x8 in1;
  vlda.conv.fp32.bf16	 cml3, [p3], #64; vldb x4, [p4, #0]
  vlda.conv.fp32.bf16	 cmh3, [p3], #64; vldb x5, [p4, #64]; padds [p4], #256
  //load tile3 8x8 in0 ;                    load tile3 8x8 in1;
  vlda.conv.fp32.bf16	 cml4, [p3], #64; vldb x0, [p4, #0]
  vlda.conv.fp32.bf16	 cmh4, [p3], #64; vldb x1, [p4, #64]
  // load output;                      
  vlda.conv.fp32.bf16	 cml2, [p2, #0]; mov	r0, #52
  vlda.conv.fp32.bf16	 cmh2, [p2, #64]; mov	r1, #53
  mov r3, #780
  // transpose in1 tile1; convert in0 tile1
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex8, dm0
  vshuffle x3, x0, x1, r1
  // convert in1 ->fp32;
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // convert tile2 in0 fp32 -> bfp16; transpose tile2
  vconv.bfp16ebs8.fp32 ex7, dm3; vshuffle x6, x4, x5, r0
  // convert in1 tile1 bfp16
  vconv.bfp16ebs8.fp32 ex0, dm1; vshuffle x7, x4, x5, r0
  // convert tile2 in1 ->fp32 -> bfp16
  vconv.fp32.bf16 cml1, x6
  vconv.fp32.bf16 cmh1, x7
  vconv.bfp16ebs8.fp32 ex1, dm1; vshuffle x2, x0, x1, r0
  
  // matrix multiplication tile1; convert tile3 in0 bfp16
  vmac.f dm2, dm2, ex8, ex0, r3; vconv.bfp16ebs8.fp32 ex8, dm4; vshuffle x3, x0, x1, r1
  // convert tile 3 in1 ->fp32  -> bfp16
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  //matrix multiplication tile 2
  vmac.f dm2, dm2, ex7, ex1, r3; vconv.bfp16ebs8.fp32 ex0, dm1
  nop
  nop
  nop
  vmac.f dm2, dm2, ex8, ex0, r3
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
