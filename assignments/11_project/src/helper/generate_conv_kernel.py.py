#!/usr/bin/env python3
"""
Generate the *optimized* AIE2p BFP16 conv kernel for N in {16,32,64} (K=64).

Structure (derived from the working src/size16/matmul_optimized.s):

  G = N/8 groups per operand; a group = 8 vst.push.576 (= 576 B = exactly
  9 x 512-bit store lines, which is why `padda [pX], m3` with m3=576 works
  here but not for the 4-push form).

  HEAD  (bundles 1..24)              : in0 group 1, pushes every 2 bundles
  UNIT u (25+20u .. 44+20u), u=0..G-2: in1 group u+1 then in0 group u+2
  TAIL  (25+20(G-1) .. 34+20(G-1))   : in1 group G

  total = 20G + 14 bundles;  `ret lr` at T+4 (= total-5).

Slot budget per 20-bundle UNIT is exactly saturated on the store port:
  8 + 8 pushes + 2 flushes + 2 `mov p2,pX` = 20.
"""

W = (22, 35, 19, 51, 16)  # v, a, b, store, scalar field widths

PUSH = "vst.push.576.conv.bfp16ebs8.fp32 {}, [p2, sf, r26]"
FLUSH = "vst.flush.512.conv [p2, sf, r26]"
VLDA = "vlda.conv.fp32.bf16 {}, [p0], #64"
VLDB = "vldb {}, [p1], #64"
VMUL = "vmul.f dm4, {}, y5, r4"

# fixed rotations, one full cycle per group
A_FILL = ["cml0", "cmh0", "cml1", "cmh1", "cml2", "cmh2", "cml3", "cmh3",
          "cml4", "cmh4", "cml0", "cmh0", "cml1", "cmh1", "cml2", "cmh2"]
B_LOAD = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
          "x8", "x9", "x0", "x1", "x2", "x3", "x4", "x5"]
V_MUL = ["y0", "y1", "y2", "y3", "y4", "y0", "y1", "y2"]
DM_PUSH = ["dm0", "dm1", "dm2", "dm3", "dm4", "dm0", "dm1", "dm2"]


class Bundle:
    __slots__ = ("v", "a", "b", "s", "m", "x", "wide", "cmt")

    def __init__(self):
        self.v, self.a, self.b = "nopv", "nopa", "nopb"
        self.s, self.m, self.x = "nops", "nopm", "nopx"
        self.wide = False   # movxm occupies slots 5+6 -> emit only 5 fields
        self.cmt = ""

    def render(self):
        f = [self.v.ljust(W[0]), self.a.ljust(W[1]), self.b.ljust(W[2]),
             self.s.ljust(W[3])]
        f.append(self.m if self.wide else self.m.ljust(W[4]))
        if not self.wide:
            f.append(self.x)
        return " ; ".join(f).rstrip() + self.cmt


def gen(N):
    assert N % 8 == 0
    G = N // 8
    total = 20 * G + 14
    bs = [Bundle() for _ in range(total + 1)]   # 1-based

    def B(i):
        return bs[i]

    # ---------------- HEAD: scalar setup + in0 group 1 ----------------
    B(1).s, B(1).m = "movs p3, p1", "mov p2, p0"
    B(2).s, B(2).m = "movs p4, p2", "mov m3, #576"
    B(3).m = "mov crrnd, #12"
    B(4).m, B(4).wide = "movxm r3, #16256", True
    B(5).m = "vbcst.16 x10, r3"
    B(6).m = "vbcst.16 x11, r3"
    B(7).m = "mov r4, #60"
    B(8).m = "mov r26, #0"

    # 8 dm0 fills at 1..16 feed 8 pushes at 9,11,..,23 (dm0 as a delay line)
    for k in range(16):
        B(1 + k).a = VLDA.format("cml0" if k % 2 == 0 else "cmh0")
    for k in range(8):
        B(9 + 2 * k).s = PUSH.format("dm0")
    B(15).cmt = " // r26 max -64 danach kaputt"
    B(17).a = "padda [p4], m3"          # p4 -> in0 group 2 base
    B(24).s = FLUSH

    # ---------------- in0 group fills (groups 2..G) ----------------
    # group g (g>=2) is pushed in unit u=g-2; its 16 fills sit at 21+20u..36+20u
    for u in range(G - 1):
        for k in range(16):
            B(21 + 20 * u + k).a = VLDA.format(A_FILL[k])

    # ---------------- in1 group loads / muls (groups 1..G) ----------------
    # group h (1-based) : vldb at 5+20(h-1)..20+20(h-1), vmul at 20+.. , push at 26+..
    for h in range(G):
        for k in range(16):
            B(5 + 20 * h + k).b = VLDB.format(B_LOAD[k])
        for k in range(8):
            B(20 + 20 * h + k).v = VMUL.format(V_MUL[k])

    # ---------------- UNIT bodies ----------------
    for u in range(G - 1):
        base = 25 + 20 * u
        # in1 group u+1
        B(base).m = "mov p2, p3"
        for k in range(8):
            B(base + 1 + k).s = PUSH.format("dm4")
        B(base + 9).s = FLUSH
        # in0 group u+2
        B(base + 10).m = "mov p2, p4"
        for k in range(8):
            B(base + 11 + k).s = PUSH.format(DM_PUSH[k])
        B(base + 19).s = FLUSH
        # pointer bumps, placed in the 4-bundle a-slot gap at base+12..base+15
        B(base + 12).a = "padda [p3], m3"          # -> in1 group u+2
        if u <= G - 3:
            B(base + 13).a = "padda [p4], m3"      # -> in0 group u+3

    # ---------------- TAIL: in1 group G ----------------
    T = 25 + 20 * (G - 1)
    B(T).m = "mov p2, p3"
    for k in range(8):
        B(T + 1 + k).s = PUSH.format("dm4")
    B(T + 9).s = FLUSH
    B(T + 4).x = "ret lr"

    hdr = ["  .section .text.conv,\"ax\",@progbits",
           "  .globl conv", "  .p2align 4",
           "  .type conv,@function", "conv:"]
    return "\n".join(hdr + [B(i).render() for i in range(1, total + 1)]) + "\n"


if __name__ == "__main__":
    import sys
    print(gen(int(sys.argv[1])), end="")
