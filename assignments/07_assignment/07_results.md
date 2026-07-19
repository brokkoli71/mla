# Submission 07: Inferring the VLIW ISA of XDNA2

In this assignment, you will infer key properties of the VLIW instruction set architecture (ISA) of the XDNA2 (AIE2P) compute tile.
You will write two simple AIE-API kernels, compile them with the Peano compiler, and systematically extract ISA properties from the generated assembly.

All kernel source files and supporting files are in `src/`.
Build artifacts go into `build/` (created by `make`).

**Prerequisites:** Run the following to enter the development environment and install torch.

```
nix develop nix-amd-npu#iron-full
iron-fhs
pip install torch                  # only once
```

---

## Task 1 — Vector-Add Kernel

The file `src/vadd.cpp` contains the function `vadd`, which operates on two 64-element BF16 input vectors and produces one 64-element BF16 output vector.

1. **Fill in** the `TODO` block and perform an element-wise vector addition.

2. **Compile** the function to assembly:

```
make asm_vadd   # → build/vadd.s
```

3. **Implement** the `verify()` function for the `vadd` kernel in `src/driver.py` and run the kernel:

```
make run_vadd   # builds xclbin for BF16 vadd and invokes driver.py
```

**Question:** What is the mnemonic used for the BF16 element-wise addition? (Inspect `build/vadd.s`.)
```asm
vadd.f
```
---

## Task 2 — Identify VLIW Slots

The XDNA2 VLIW instruction word has six functional-unit slots.
Each slot is either occupied by an operation, filled with a NOP mnemonic specific to that slot, or kept empty.

Below are the first two VLIW instructions from `build/vadd.s`:

```asm
vlda.conv.fp32.bf16 cml0, [p0, #0]; nopb; nops; nopxm; nopv
vlda.conv.fp32.bf16 cmh0, [p0, #64]; nopx
```

Inspect the instructions and **fill in** the table.

(erklärung: occupied -> hat eine anweisung e.g. vlda, nop)

| Functional Unit     | Slot   | NOP mnemonic | Occupied in the second instruction? |
|---------------------|--------|--------------|-------------------------------------|
| Vector Unit         | V      | nopv         |   no                                |
| Load Unit A         | A      | nopa         |     yes                             |
| Load Unit B         | B      | nopb         |     no                              |
| Store Unit          | S      | nops         |       no                            |
| Scalar/Control Unit | X (XM) | nopx         |      yes                            |
| Movement Unit       | M (XM) | nopm         |       no                            |

---

## Task 3 — Identify Instructions and Register Classes per Slot

**Hints:**
- Mnemonic prefix/suffix indicates the slot.
- Register name prefix indicates class: `p` → pointer register; `r` → scalar register; `x`/`y` → vector register; `dm`/`cm`/`bm` → accumulator register.

1. **Fill in** the instruction table:

| Instruction                          | Slot | Short description (optional) |
|--------------------------------------|------|------------------------------|
| `vlda.conv.fp32.bf16 cml0, [p0, #0]` | A    | load p0 into cml0 (fp32 to bf16)|
| `movx r6, #1`                        | X    | write 1 to register 6        |
| `vldb x1, [p1, #0]`                  | B    | load p1 to vector register 1 |
| `vmov bmhl2, bmhh4`                  | V    | vector move bmhh4 to bmhl2   |
| `mova r0, #60`                       | M/A? | Write 60 to register 0       |
| `vadd.f dm0, dm0, dm1, r0`           | V    | add dm1 onto dm0 (60 is configuration) |
| `ret lr`                             | X    |     return                   |
| `mov p1, p4`                         | M    |     move p4 to p1            |
| `vst.conv.bf16.fp32 cml0, [p2, #0]`  | S    |    store cml0 to p2          |

2. **Fill in** the register-class table:

| Slot | Register classes (dst / src) | Example registers |
|------|------------------------------|-------------------|
| V    |  dm,cm,bm                    | dm0               |
| A    |  from p to dm,cm,bm,x,y      | p0                |
| B    |  from p to x,y               | p0                |
| S    |  from dm,cm,bm,x,y to p      | dm0               |
| X    |  x,y                         | x1                |
| M    |  p,r                         | r0                |
| XM   |  x,y,p,r                     | x1                |

| Slot | Register classes (dst / src) | Example registers |
|------|------------------------------|-------------------|
| V    | Vector Reg, Acuumulator reg  | x3, y2, wl7, cm0  |
| A    | Vector, Accum. / p           | p1, x7, cm1       |
| B    | Vector, Accum. / p           | p1, x7, bml0      |
| S    | p / Vector, Accum.           | p1, x7, bmh0 |
| X    | Scalar                       | r11               |
| M    | Scalar, vector               | r12, x6           |
| XM   | Scalar, vector               | r5, y3            |

---

## Task 4 — Infer Operation Latencies

The XDNA2 compute tile has **no stall logic**: if a consumer instruction reads a register before the producing instruction has written it, the result is undefined.
The compiler (and the programmer, in hand-scheduled assembly) must insert NOP instructions to respect the latency of every operation.

**Counting rule:** Latency is the number of cycles from the producing instruction (cycle 1) to the first dependent instruction (exclusive).
A NOP between producer and consumer counts as one cycle.

Using `build/vadd.s`, **fill in** the latency table:

| Instruction | Output register | First dependent instruction | Cycles apart | Latency |
|-------------|-----------------|-----------------------------|:------------:|:-------:|
| `mova`      | r0              |  vadd.f                     | 1            | 1       |
| `vadd.f`    | dm0             |  vst.conv.bf16.fp32         | 6            | 6       |

---

## Task 5 — Write a BF16 Vector-Add Kernel in Hand-Scheduled Assembly

Write an assembly kernel that computes `C = A + B + B`.
Using the ISA knowledge gained in Tasks 1–4, manually schedule the vadd operation into VLIW instructions.

1. **Replace** the `TODO` comments with actual instructions in `src/custom_vadd.s`.

   Observe the following constraints:

   - Insert only as many NOP cycles as strictly required by the latencies.
   - You may use the five `ret` delay slots to place useful instructions (e.g., the store).

2. **Assemble** your kernel:

```
make obj_custom_vadd   # → build/custom_vadd.o
```

The target must assemble without errors.

3. **Implement** the `verify()` function for the `custom_vadd` kernel in `src/driver.py` and run your kernel:

```
make run_custom_vadd     # builds xclbin for BF16 custom_vadd and invokes driver.py
```
**Question:**
How many VLIW cycles does your hand-scheduled kernel take?
Is this the fewest possible number of cycles?

- We load B before A, then we can start the addition of B to itself as soon as B is loaded, without waiting for A to be loaded. 
- Storing the result can be done in the last delay slot of the return instruction -> less nops before `ret lr`
- It needs 16 cycles. We think it's the fewest possible.

```{literalinclude} src/custom_vadd.s
:language: asm
:start-after: custom_vadd:
:end-before: .Lfunc_end0:
```

---

## Task 6 — MAC Kernel (optional)

The file `src/matmul.cpp` contains a function `matmul` for an 8×8×8 BF16 matrix multiplication. The accumulator uses FP32 as its datatype.

**Compile** the matmul kernel with the two provided Makefile targets:

```
make asm_matmul        # → build/matmul_normal.s
make asm_matmul_bfp16  # → build/matmul_bfp16.s
```

The second target adds `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`, which instructs the AIE-API to map the 8×8×8 BF16 matmul to a single native BFP16 multiply-accumulate instruction instead of using multiple BF16 operations.

**Questions:**

a) How many instructions does each mode produce?

b) What does the `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` flag change about how the matmul is compiled?

c) What are the performance implications of using BFP16 mode vs. normal mode?

