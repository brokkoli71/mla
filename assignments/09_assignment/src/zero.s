  .file "zero.s"
  .section .text.zero,"ax",@progbits
  .globl zero
  .p2align 4
  .type zero,@function
zero:
  // Sets 512 bytes to zero
  mov r0, #0
  vbcst.16 x0, r0
  ret lr
  vst x0, [p0], #64                   // Delay Slot 5
  vst x0, [p0], #64                   // Delay Slot 4
  vst x0, [p0], #64                   // Delay Slot 3
  vst x0, [p0], #64                   // Delay Slot 2
  nop                                 // Delay Slot 1
.Lfunc_end0:
  .size zero, .Lfunc_end0-zero
