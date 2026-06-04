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
// ca. 184 cycles in the end

// TODO: implement tensor kernel
  mov p3, p0 
  mov m0, #1024
  padds [p3], m0
  #padds [p3], #256
  #padds [p3], #256
  #padds [p3], #256
  #padds [p3], #256
  #mov p5, p2
  // load output tile p1 q1;                      
  vlda.conv.fp32.bf16	 cml2, [p2, #0] 
  vlda.conv.fp32.bf16	 cmh2, [p2, #64]
  vlda.conv.fp32.bf16	 cml4, [p2, #256]
  vlda.conv.fp32.bf16	 cmh4, [p2, #320]
  //load 8x8 in0 ;                    load 8x8 in1; copy base pointer
  vlda.conv.fp32.bf16	 cml0, [p0], #64; vldb x0, [p1, #0]; mov p4, p1
  vlda.conv.fp32.bf16	 cmh0, [p0], #64; vldb x1, [p1, #64]; padds [p4], #256
  // load output tile p2 q1; 
  vlda.conv.fp32.bf16	 cml3, [p3], #64; mov	r0, #52
  vlda.conv.fp32.bf16	 cmh3, [p3], #64; mov	r1, #53
    //load 2nd 8x8 in0 ;                    load 2nd 8x8 in1;
  vlda.conv.fp32.bf16	 cml0, [p0], #64; vldb x0, [p4], #64   ; mov r3, #780
  vlda.conv.fp32.bf16	 cmh0, [p0], #64; vldb x1, [p4], #192
  // load in0 tile  p2
  vlda.conv.fp32.bf16	 cml3, [p3], #64
  vlda.conv.fp32.bf16	 cmh3, [p3], #64
                                                                                              //load tile3 8x8 in0 ;                    load tile3 8x8 in1;
                                                                                              vlda.conv.fp32.bf16	 cml0, [p0], #64; vldb x0, [p4], #64
  // transpose in1 tile1; convert in0 tile1 bfp16
  vshuffle x2, x0, x1, r0;  vconv.bfp16ebs8.fp32 ex10, dm0;                                   vlda.conv.fp32.bf16	 cmh0, [p0], #64; vldb x1, [p4], #192                  
  vshuffle x3, x0, x1, r1                                                                                                                                              
  // convert in1 ->fp32;                                                                      load tile4                                                               
  vconv.fp32.bf16 cml1, x2                                                                                                             
  vconv.fp32.bf16 cmh1, x3;                                                                   vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4], #64
  // transpose in1 tile2; convert tile2 in0 fp32 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex9, dm0;                                     vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4], #192   
  // still transpose tile2; convert in1 tile1 bfp16
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1
  // convert tile2 in1 ->fp32;convert in0 tile1 p2                                            load tile5
  vconv.fp32.bf16 cml1, x2   
  vconv.fp32.bf16 cmh1, x3; vconv.bfp16ebs8.fp32 ex8, dm3;                                    vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4], #64
  // transpose in1 tile3; convert tile3 in0 ->bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0;                                    vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4], #192
  //                     ; convert tile2 in1 bfp16;      ; matrix multiplication tile1
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex10, ex11, r3
  // convert tile 3 in1 ->fp32;                            matrix multiplication tile1 p2     load tile6
  vconv.fp32.bf16 cml1, x2                                    
  vconv.fp32.bf16 cmh1, x3;                                vmac.f dm4, dm4, ex8 , ex11, r3;   vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4], #64
  // transpose in1 tile4; convert tile4 in0 -> bfp16 
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex9, dm0;                                     vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4], #192
  //                     ; convert tile3 in1 ->bfp16;    ; matrix multiplication tile 2
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex9, ex11, r3
  // convert tile4 in1 ->fp32                                                                 load tile7
  vconv.fp32.bf16 cml1, x2                                                                    
  vconv.fp32.bf16 cmh1, x3;                                                                   vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4], #64 
  // transpose in1 tile5; convert tile5 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0;                                    vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4], #192
  //                     ; convert tile4 in1 -> bfp16    ; matrix mul tile3
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex10, ex11, r3
  // convert in1 tile 5 ->fp32;                                                               load tile8
  vconv.fp32.bf16 cml1, x2                                                                     
  vconv.fp32.bf16 cmh1, x3;                                                                   vlda.conv.fp32.bf16	cml0, [p0], #64; vldb x0, [p4], #64
  // transpose in1 tile6; convert tile6 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex9, dm0;                                     vlda.conv.fp32.bf16	cmh0, [p0], #64; vldb x1, [p4], #192
  //                     ; convert tile5 in1 -> bfp16    ; matrix mul tile4;
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex9, ex11, r3
  // convert in1 tile 6 ->fp32; 
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // transpose in1 tile7; convert tile7 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0
  //                     ; convert tile6 in1 -> bfp16    ; matrix mul tile5;
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex10, ex11, r3
  // convert in1 tile 7 ->fp32; 
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // transpose in1 tile8; convert tile8 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex9, dm0
  //                     ; convert tile7 in1 -> bfp16    ; matrix mul tile6;
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex9, ex11, r3
    // convert in1 tile 8 ->fp32; 
  vconv.fp32.bf16 cml1, x2
  vconv.fp32.bf16 cmh1, x3
  // transpose in1 tile9; convert tile9 in0 -> bfp16
  vshuffle x2, x0, x1, r0; vconv.bfp16ebs8.fp32 ex10, dm0
  //                     ; convert tile8 in1 -> bfp16    ; matrix mul tile7;
  vshuffle x3, x0, x1, r1; vconv.bfp16ebs8.fp32 ex11, dm1; vmac.f dm2, dm2, ex10, ex11, r3
  nop
  nop
  nop // matmul til8
  vmac.f dm2, dm2, ex9, ex11, r3
  nop
  nop
  nop
  nop
  nop
  vst.conv.bf16.fp32 cml2, [p2, #0]
  vst.conv.bf16.fp32 cmh2, [p2, #64]  
  vst.conv.bf16.fp32 cml4, [p2, #256]
  vst.conv.bf16.fp32 cmh4, [p2, #320]
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
  nop                                 // Delay Slot 3
  nop                                 // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size matmul, .Lfunc_end0-matmul
