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
  #mov p3, p0
  #mov p5, p2
  //load 8x8 in0 ;                    load 8x8 in1; copy base pointer
  vlda.conv.fp32.bf16	 cml0, [p0], #64; vldb x0, [p1, #0]; mov p4, p1 //padds [p3], #128 
  vlda.conv.fp32.bf16	 cmh0, [p0], #64; vldb x1, [p1, #64]; padds [p4], #256
    //load 2nd 8x8 in0 ;                    load 2nd 8x8 in1;
  vlda.conv.fp32.bf16	 cml3, [p0], #64; vldb x0, [p4, #0]
  vlda.conv.fp32.bf16	 cmh3, [p0], #64; vldb x1, [p4, #64]; padds [p4], #256
  // load output;                      
  vlda.conv.fp32.bf16	 cml2, [p2, #0]; mov	r0, #52
  vlda.conv.fp32.bf16	 cmh2, [p2, #64]; mov	r1, #53
  mov r3, #780
  //load tile3 8x8 in0 ;                    load tile3 8x8 in1;
  vlda.conv.fp32.bf16	 cml4, [p0], #64; vldb x0, [p4, #0]
  vlda.conv.fp32.bf16	 cmh4, [p0], #64; vldb x1, [p4, #64]; padds [p4], #256
  // transpose in1 tile1; convert in0 tile1
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0
  vshuffle x3, x0, x1, r1
  // convert in1 ->fp32   ; load tile4
  vconv.fp32.bf16 cml1, x2;  vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4, #0]
  vconv.fp32.bf16 cmh1, x3;  vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4, #64]; padds [p4], #256
  // convert in1 tile1 bfp16; ; transpose tile2
  vconv.bfp16ebs8.fp32 ex11, dm1; vshuffle x2, x0, x1, r0
  // convert tile2 in0 fp32 -> bfp16; still transpose tile2
  vconv.bfp16ebs8.fp32 ex10, dm3;  vshuffle x3, x0, x1, r0
  // convert tile2 in1 ->fp32 
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // matrix multiplication tile1; convert tile2 in1 bfp16; transpose in1 tile 3
  vmac.f dm2, dm2, ex10, ex11, r3; vconv.bfp16ebs8.fp32 ex11, dm1; vshuffle x2, x0, x1, r0
  // convert tile3 in0 ->bfp16; still transpose in1 tile3
  vconv.bfp16ebs8.fp32 ex11, dm4; vshuffle x3, x4, x5, r1
  // convert tile 3 in1 ->fp32
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  //matrix multiplication tile 2; convert tile3 in1 ->bfp16; transpose tile4 in1
  vmac.f dm2, dm2, ex10, ex11, r3; vconv.bfp16ebs8.fp32 ex10, dm1; vshuffle x2, x0, x1, r0
  // still transpose tile4 in1; convert tile4 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0
  // convert tile4 in1 ->fp32
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // matrix mul tile3; convert tile4 in1 -> bfp16
  vmac.f dm2, dm2, ex10, ex11, r3;  vconv.bfp16ebs8.fp32 ex11, dm1
  nop
  nop
  nop
  // matrix mul tile4;
  vmac.f dm2, dm2, ex10, ex11, r3
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
