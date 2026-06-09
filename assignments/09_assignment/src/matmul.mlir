module { // "wir sind in einem mla modul"
  aie.device(npu2) {
    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) attributes {link_with = "matmul.o"}
    func.func private @zero(memref<2x2x8x8xbf16>) attributes {link_with = "zero.o"}
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    //objectfifo queues beschreiben was wie verbunden wird
    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])
    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0] -> [@out_L2L3_0]([] [])
    %core_0_2 = aie.core(%tile_0_2) { // funktion, die auf dem compute tile läuft
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 { //while max int for preloading kind of (?)
        %abmax = arith.constant 128 : index
        scf.for %arg1 = %c0 to %abmax step %c1 { //for a*b
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %cmax = arith.constant 16 : index
          scf.for %arg2 = %c0 to %cmax step %c1 { //for c
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1) // "ich habe gelesen, können wieder überschrieben werden"
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}
    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {
      // TODO: Add needed data movement operations; adapt offsets, sizes, and strides accordingly.
      // Note: There are only 16 buffer descriptors (ids) on the shim tile; synchronize before reuse.
      //       dma_wait will synchronize with the first issued corresponding data movement.
      
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 1, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[1, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}
 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 2, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[2, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 3, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[3, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 4, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[4, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 5, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[5, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 6, 0, 0][16, 1, 16, 16][2048, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[6, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

 
      // dimension: a, b, pm, qn
      aiex.npu.dma_memcpy_nd(%arg2[0, 7, 0, 0][16, 1, 16, 16][0, 16, 128, 1]) {id = 0 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      // dimension: a, c, pm, rk
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][16, 16, 16, 64][16384, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      // dimension: b, c, rk, qn
      aiex.npu.dma_memcpy_nd(%arg1[7, 0, 0, 0][1, 16, 64, 16][16, 8192, 128, 1]) {id = 2 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @in0_L3L2_0}
      aiex.npu.dma_wait {symbol = @in1_L3L2_0}

  
    }
  }
}
