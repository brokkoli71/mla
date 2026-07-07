"""
Generate .mlir instances for the parameterized tensor kernel
`tensor_kernel_m1xn1xk1_bf16_bf16_bf16` at a given (m2, n2, k1) tiling.

The kernel object is parameterized (m2/n2/k1 in registers), but each .mlir bakes
in concrete objectfifo memref shapes, zeroing-loop bounds, the kernel's integer
args, and the shim DMA byte sizes -- so we need one .mlir per shape. This emits
both the plain instance and a benchmark variant (compute call wrapped in an inner
scf.for of `reps` iterations so one host launch drives many on-chip iterations).

    M = m2 * 2 * 8,   N = n2 * 2 * 8,   K = k1 * 8
    constraints: m2*n2 >= 2,  k1 >= 6 and even

Usage:
    python3 gen_mlir.py <m2> <n2> <k1> [reps]   > instance.mlir
    (reps omitted -> plain instance; reps given -> benchmark variant)
"""

import sys

OBJ = "tensor_kernel_m1xn1xk1_bf16_bf16_bf16"


def gen(m2: int, n2: int, k1: int, reps: int | None = None) -> str:
    in0 = f"{m2}x{k1}x2x8x8"
    in1 = f"{n2}x{k1}x2x8x8"
    out = f"{m2}x{n2}x2x2x8x8"
    n_in0 = m2 * k1 * 2 * 8 * 8
    n_in1 = n2 * k1 * 2 * 8 * 8
    n_out = m2 * n2 * 2 * 2 * 8 * 8

    # inline zeroing of the whole output tile (once per host acquire)
    zero = f"""        scf.for %z0 = %c0 to %cbm2 step %c1 {{
          scf.for %z1 = %c0 to %cbn2 step %c1 {{
            scf.for %z2 = %c0 to %c2 step %c1 {{
              scf.for %z3 = %c0 to %c2 step %c1 {{
                scf.for %z4 = %c0 to %c8 step %c1 {{
                  scf.for %z5 = %c0 to %c8 step %c1 {{
                    memref.store %cst, %1[%z0, %z1, %z2, %z3, %z4, %z5] : memref<{out}xbf16>
                  }}
                }}
              }}
            }}
          }}
        }}"""

    call = (f"        func.call @{OBJ}(%3, %5, %1, %cm2, %cn2, %ck1) : "
            f"(memref<{in0}xbf16>, memref<{in1}xbf16>, memref<{out}xbf16>, "
            f"i32, i32, i32) -> ()")
    if reps is not None:
        compute = (f"        scf.for %r = %c0 to %creps step %c1 {{\n"
                   f"  {call}\n"
                   f"        }}")
        reps_const = f"        %creps = arith.constant {reps} : index\n"
    else:
        compute = call
        reps_const = ""

    return f"""module {{
  aie.device(npu2_1col) {{
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {{%tile_0_2}}, 2 : i32) : !aie.objectfifo<memref<{in0}xbf16>>
    aie.objectfifo @in1(%shim_noc_tile_0_0, {{%tile_0_2}}, 2 : i32) : !aie.objectfifo<memref<{in1}xbf16>>
    aie.objectfifo @out0(%tile_0_2, {{%shim_noc_tile_0_0}}, 1 : i32) : !aie.objectfifo<memref<{out}xbf16>>
    func.func private @{OBJ}(memref<{in0}xbf16>, memref<{in1}xbf16>, memref<{out}xbf16>, i32, i32, i32) attributes {{link_with = "{OBJ}.o"}}
    %core_0_2 = aie.core(%tile_0_2) {{
      %c0 = arith.constant 0 : index
      %cinf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %cbm2 = arith.constant {m2} : index
      %cbn2 = arith.constant {n2} : index
      %cm2 = arith.constant {m2} : i32
      %cn2 = arith.constant {n2} : i32
      %ck1 = arith.constant {k1} : i32
      %cst = arith.constant 0.000000e+00 : bf16
{reps_const}      scf.for %arg0 = %c0 to %cinf step %c1 {{
        %0 = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<{out}xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<{out}xbf16>> -> memref<{out}xbf16>
{zero}
        %2 = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<{in0}xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<{in0}xbf16>> -> memref<{in0}xbf16>
        %4 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<{in1}xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<{in1}xbf16>> -> memref<{in1}xbf16>
{compute}
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }}
      aie.end
    }}
    aie.runtime_sequence(%arg0: memref<{in0}xbf16>, %arg1: memref<{in1}xbf16>, %arg2: memref<{out}xbf16>) {{
      %0 = aiex.dma_configure_task_for @in0 {{
        aie.dma_bd(%arg0 : memref<{in0}xbf16>, 0, {n_in0}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_in0}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in1 {{
        aie.dma_bd(%arg1 : memref<{in1}xbf16>, 0, {n_in1}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_in1}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out0 {{
        aie.dma_bd(%arg2 : memref<{out}xbf16>, 0, {n_out}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {n_out}, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }}
  }}
}}
"""


if __name__ == "__main__":
    m2, n2, k1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else None
    sys.stdout.write(gen(m2, n2, k1, reps))
