  .file "custom_vadd.s"
  .section .text.custom_vadd,"ax",@progbits
  .globl custom_vadd
  .p2align 4
  .type custom_vadd,@function
custom_vadd:
// Computes C = A + B + B
// Calling convention: p0 = ptr_in0, p1 = ptr_in1, p2 = ptr_out
  // load 128 bytes from ptr_in0 (p0) into a accumulator register (A-slot)
  vlda.conv.fp32.bf16	 cml0, [p0, #0];		nopb	;		nops	;		nopxm	;		nopv	
  vlda.conv.fp32.bf16	 cmh0, [p0, #64];		nopx	
  // load 128 bytes from ptr_in1 (p1) into a accumulator register (A-slot)
  vlda.conv.fp32.bf16	 cml1, [p1, #0]
	vlda.conv.fp32.bf16	 cmh1, [p1, #64]
  // insert NOPs to satisfy load latency
  nop	
	nop	
	mova	r0, #60
  // perform two BF16 elementwise add using the mnemonic found in build/vadd.s (V-slot)
  vadd.f	dm0, dm0, dm1, r0
  nop
  nop
	vadd.f	dm0, dm0, dm1, r0
  // insert NOPs to satisfy add latency
  nop	
	nop	
  // store result to ptr_out (p2) using a store instruction (S-slot)  
  ret lr
  nop                                 // Delay Slot 5
  nop                                 // Delay Slot 4
	vst.conv.bf16.fp32	 cml0, [p2, #0] //  Delay Slot 3
	vst.conv.bf16.fp32	 cmh0, [p2, #64] //  Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size custom_vadd, .Lfunc_end0-custom_vadd
