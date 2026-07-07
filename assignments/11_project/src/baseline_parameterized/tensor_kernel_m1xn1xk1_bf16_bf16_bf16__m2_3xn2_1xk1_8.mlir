module {
  aie.device(npu2_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<3x8x2x8x8xbf16>> 
    aie.objectfifo @in1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1x8x2x8x8xbf16>> 
    aie.objectfifo @out0(%tile_0_2, {%shim_noc_tile_0_0}, 1 : i32) : !aie.objectfifo<memref<3x1x2x2x8x8xbf16>> 
    func.func private @tensor_kernel_m1xn1xk1_bf16_bf16_bf16(memref<3x8x2x8x8xbf16>, memref<1x8x2x8x8xbf16>, memref<3x1x2x2x8x8xbf16>, i32, i32, i32) attributes {link_with = "tensor_kernel_m1xn1xk1_bf16_bf16_bf16.o"}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<3x1x2x2x8x8xbf16>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<3x1x2x2x8x8xbf16>> -> memref<3x1x2x2x8x8xbf16>
        %c0_0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c3 step %c1_1 {
          %c0_2 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c1_4 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c1_3 step %c1_4 {
            %c0_5 = arith.constant 0 : index
            %c2 = arith.constant 2 : index
            %c1_6 = arith.constant 1 : index
            scf.for %arg3 = %c0_5 to %c2 step %c1_6 {
              %c0_7 = arith.constant 0 : index
              %c2_8 = arith.constant 2 : index
              %c1_9 = arith.constant 1 : index
              scf.for %arg4 = %c0_7 to %c2_8 step %c1_9 {
                %c0_10 = arith.constant 0 : index
                %c8 = arith.constant 8 : index
                %c1_11 = arith.constant 1 : index
                scf.for %arg5 = %c0_10 to %c8 step %c1_11 {
                  %c0_12 = arith.constant 0 : index
                  %c8_13 = arith.constant 8 : index
                  %c1_14 = arith.constant 1 : index
                  scf.for %arg6 = %c0_12 to %c8_13 step %c1_14 {
                    %cst = arith.constant 0.000000e+00 : bf16
                    memref.store %cst, %1[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6] : memref<3x1x2x2x8x8xbf16>
                  }
                }
              }
            }
          }
        }
        %2 = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<3x8x2x8x8xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<3x8x2x8x8xbf16>> -> memref<3x8x2x8x8xbf16>
        %4 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<1x8x2x8x8xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1x8x2x8x8xbf16>> -> memref<1x8x2x8x8xbf16>
        %c3_i32 = arith.constant 3 : i32
        %c1_i32 = arith.constant 1 : i32
        %c8_i32 = arith.constant 8 : i32
        func.call @tensor_kernel_m1xn1xk1_bf16_bf16_bf16(%3, %5, %1, %c3_i32, %c1_i32, %c8_i32) : (memref<3x8x2x8x8xbf16>, memref<1x8x2x8x8xbf16>, memref<3x1x2x2x8x8xbf16>, i32, i32, i32) -> ()
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<3x8x2x8x8xbf16>, %arg1: memref<1x8x2x8x8xbf16>, %arg2: memref<3x1x2x2x8x8xbf16>) {
      %0 = aiex.dma_configure_task_for @in0 {
        aie.dma_bd(%arg0 : memref<3x8x2x8x8xbf16>, 0, 3072, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 3072, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg1 : memref<1x8x2x8x8xbf16>, 0, 1024, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1024, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out0 {
        aie.dma_bd(%arg2 : memref<3x1x2x2x8x8xbf16>, 0, 768, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 768, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}
