  .file "matmul.s"
  .section .text.matmul,"ax",@progbits
  .globl matmul
  .p2align 4
  .type matmul,@function
matmul:
// Computes out += in0 * in1
// L1 tensor views:
//   p=2, q=2, r=8, m=8, n=8, k=8
//   in0: prmk (p0) -> ex10, ex11
//   in1: rqkn (p1) -> ex0, ex1
//   out: pqmn (p2)
  // load inputs in0
  # 1 q-stride = 8 for k, 8 for m, 8 for r, 2 byte for bf16 = 1024
  vlda.conv.fp32.bf16	 cml0, [p0,#0]; mov p4, p0
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; mova	r0, #52; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; mova	r1, #53; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  vlda.conv.fp32.bf16	 cml4, [p3], #64 // only in first loop
  vlda.conv.fp32.bf16	 cmh4, [p3], #64 // only in first loop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  vlda.conv.fp32.bf16	 cml1, [p3], #64 // only in first loop
  vlda.conv.fp32.bf16	 cmh1, [p3], #64 // only in first loop
  vlda.conv.fp32.bf16	 cml2, [p3], #64 // only in first loop
  vlda.conv.fp32.bf16	 cmh2, [p3], #64 // only in first loop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  vlda.conv.fp32.bf16	 cml3, [p3], #64 // only in first loop
  vlda.conv.fp32.bf16	 cmh3, [p3], #64 // only in first loop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5
  vmac.f dm1, dm1, ex10, ex1, r5; mov p4, p0 // reset p4 for later use
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #128 // pad p4 for next iteration
  vmac.f dm3, dm3, ex11, ex1, r5
  nop

///////////////////////////////////////////// r=1
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop

///////////////////////////////////////////// r=2
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop
///////////////////////////////////////////// r=3
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop

///////////////////////////////////////////// r=4
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop
///////////////////////////////////////////// r=5
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop
///////////////////////////////////////////// r=6
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop
///////////////////////////////////////////// r=7
  // load inputs in0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64]; padds [p1], #256
  //r0, r1 for transposing
  //r3 for ones
  // create ones in y4 (x8, x9)
  vldb x0, [p1, #0];padds [p4], #256; movxm r3, #16256
  vldb x1, [p1, #64];padds [p4], #256; vbcst.16 x8, r3
  vldb x2, [p1, #128];padds [p4], #256; mova r4, #60
  vldb x3, [p1, #192];padds [p4], #256; vmov x9, x8
  nop
  nop
  vconv.bfp16ebs8.fp32 ex10, dm0
  vlda.conv.fp32.bf16	 cml0, [p4,#0]
  vlda.conv.fp32.bf16	 cmh0, [p4,#64];   mov p3, p2   // copy init value of p2 to p3 for later store
  vshuffle x4, x0, x1, r0
  vshuffle x5, x0, x1, r1
  vshuffle x6, x2, x3, r0
  vshuffle x7, x2, x3, r1
  nop
  nop
  vconv.bfp16ebs8.fp32 ex11, dm0
  // BF16 -> FP32 -> BFP16 of in1
  vmul.f dm0, y2, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex0, dm0
  vmul.f dm0, y3, y4, r4
  nop
  nop
  nop
  nop
  nop
  vconv.bfp16ebs8.fp32 ex1, dm0 //inputs loaded after 32 cycles of ocupying dm0
  
  // all inputs are in ex, outputs can be loaded to dm
  // load outputs
  // matrix multiplication
  nop
  mova r5, #780
  vmac.f dm4, dm4, ex10, ex0, r5; padds [p4], #-256 // pad p4 back for next iteration
  vmac.f dm1, dm1, ex10, ex1, r5; padds [p4], #-384
  vmac.f dm2, dm2, ex11, ex0, r5; padds [p4], #-256
  vmac.f dm3, dm3, ex11, ex1, r5
  nop



  nop
  // store outputs
  vst.conv.bf16.fp32 cml4, [p2], #64
  vst.conv.bf16.fp32 cmh4, [p2], #64
  vst.conv.bf16.fp32 cml1, [p2], #64
  vst.conv.bf16.fp32 cmh1, [p2], #64
  vst.conv.bf16.fp32 cml2, [p2], #64
  vst.conv.bf16.fp32 cmh2, [p2], #64
  vst.conv.bf16.fp32 cml3, [p2], #64
  vst.conv.bf16.fp32 cmh3, [p2], #64
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul