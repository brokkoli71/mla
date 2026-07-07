module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2x8x2x8x8xbf16>>
    aie.objectfifo @in1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2x8x2x8x8xbf16>>
    aie.objectfifo @out0(%tile_0_2, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<2x2x2x2x8x8xbf16>>
    func.func private @tensor_kernel_m1xn1xk1_bf16_bf16_bf16(memref<2x8x2x8x8xbf16>, memref<2x8x2x8x8xbf16>, memref<2x2x2x2x8x8xbf16>, i32, i32, i32) attributes {link_with = "tensor_kernel_m1xn1xk1_bf16_bf16_bf16.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cinf = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %cbm2 = arith.constant 2 : index
      %cbn2 = arith.constant 2 : index
      %cm2 = arith.constant 2 : i32
      %cn2 = arith.constant 2 : i32
      %ck1 = arith.constant 8 : i32
      %cst = arith.constant 0.000000e+00 : bf16
        %creps = arith.constant 100000 : index
      scf.for %arg0 = %c0 to %cinf step %c1 {
        %0 = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<2x2x2x2x8x8xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<2x2x2x2x8x8xbf16>> -> memref<2x2x2x2x8x8xbf16>
        scf.for %z0 = %c0 to %cbm2 step %c1 {
          scf.for %z1 = %c0 to %cbn2 step %c1 {
            scf.for %z2 = %c0 to %c2 step %c1 {
              scf.for %z3 = %c0 to %c2 step %c1 {
                scf.for %z4 = %c0 to %c8 step %c1 {
                  scf.for %z5 = %c0 to %c8 step %c1 {
                    memref.store %cst, %1[%z0, %z1, %z2, %z3, %z4, %z5] : memref<2x2x2x2x8x8xbf16>
                  }
                }
              }
            }
          }
        }
        %2 = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<2x8x2x8x8xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<2x8x2x8x8xbf16>> -> memref<2x8x2x8x8xbf16>
        %4 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<2x8x2x8x8xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<2x8x2x8x8xbf16>> -> memref<2x8x2x8x8xbf16>
        scf.for %r = %c0 to %creps step %c1 {
          func.call @tensor_kernel_m1xn1xk1_bf16_bf16_bf16(%3, %5, %1, %cm2, %cn2, %ck1) : (memref<2x8x2x8x8xbf16>, memref<2x8x2x8x8xbf16>, memref<2x2x2x2x8x8xbf16>, i32, i32, i32) -> ()
        }
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<2x8x2x8x8xbf16>, %arg1: memref<2x8x2x8x8xbf16>, %arg2: memref<2x2x2x2x8x8xbf16>) {
      %0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%arg0 : memref<2x8x2x8x8xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg1 : memref<2x8x2x8x8xbf16>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%arg2 : memref<2x2x2x2x8x8xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}
