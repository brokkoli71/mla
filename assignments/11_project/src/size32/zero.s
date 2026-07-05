  .file "zero.s"
  .section .text.zero,"ax",@progbits
  .globl zero
  .p2align 4
  .type zero,@function
zero:
  // size32 output is 4x4x8x8 bf16 = 1024 elements = 2048 bytes = 32 * 64 bytes
  mov r0, #0
  vbcst.16 x0, r0
  nop
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64
  vst x0, [p0], #64; ret lr
  vst x0, [p0], #64                   // Delay Slot 5
  vst x0, [p0], #64                   // Delay Slot 4
  vst x0, [p0], #64                   // Delay Slot 3
  vst x0, [p0], #64                   // Delay Slot 2
  vst x0, [p0], #64                   // Delay Slot 1
.Lfunc_end0:
  .size zero, .Lfunc_end0-zero
