module {
  aie.device(npu2) {
    func.func private @vadd(memref<8x8xbf16>, memref<8x8xbf16>, memref<8x8xbf16>) attributes {link_with = "vadd.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 32>, <size = 1, stride = 8>, <size = 4, stride = 8>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 1, stride = 64>, <size = 1, stride = 8>, <size = 8, stride = 8>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])
    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 32>, <size = 4, stride = 8>, <size = 1, stride = 32>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0] -> [@out_L2L3_0]([] [])
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<8x8xbf16>>
        %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<8x8xbf16>> -> memref<8x8xbf16>
        %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x8xbf16>>
        %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<8x8xbf16>> -> memref<8x8xbf16>
        %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x8xbf16>>
        %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x8xbf16>> -> memref<8x8xbf16>
        func.call @vadd(%in0, %in1, %out) : (memref<8x8xbf16>, memref<8x8xbf16>, memref<8x8xbf16>) -> ()
        aie.objectfifo.release @in0_L2L1_0(Consume, 1)
        aie.objectfifo.release @in1_L2L1_0(Consume, 1)
        aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
      }
      aie.end
    } {stack_size = 3328 : i32}
    aie.runtime_sequence(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %arg2: memref<64xbf16>) {
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 8, 8][64, 8, 8, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<64xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 8, 8][0, 8, 8, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<64xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 8, 8][8, 64, 8, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<64xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
    }
  }
}
