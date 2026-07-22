module {
  aie.device(npu2) {
    func.func private @matmul(memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) attributes {link_with = "matmul.o"}
    func.func private @zero(memref<2x2x8x8xbf16>) attributes {link_with = "zero.o"}
    
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)

    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %mem_tile_6_1 = aie.tile(6, 1)
    %mem_tile_7_1 = aie.tile(7, 1)

    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_2 = aie.tile(5, 2)
    %tile_6_2 = aie.tile(6, 2)
    %tile_7_2 = aie.tile(7, 2)

    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_3 = aie.tile(4, 3)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_3 = aie.tile(7, 3)
    
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_4 = aie.tile(7, 4)
    
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_5 = aie.tile(3, 5)
    %tile_4_5 = aie.tile(4, 5)
    %tile_5_5 = aie.tile(5, 5)
    %tile_6_5 = aie.tile(6, 5)
    %tile_7_5 = aie.tile(7, 5)

    
    aie.objectfifo @in0_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_4(%shim_noc_tile_4_0, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_5(%shim_noc_tile_5_0, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_6(%shim_noc_tile_6_0, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>
    aie.objectfifo @in0_L3L2_7(%shim_noc_tile_7_0, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<16x64xbf16>>

    aie.objectfifo @in0_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_3_2, %tile_3_3, %tile_3_4, %tile_3_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_4(%mem_tile_4_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_4_2, %tile_4_3, %tile_4_4, %tile_4_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_5(%mem_tile_5_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_5_2, %tile_5_3, %tile_5_4, %tile_5_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_6(%mem_tile_6_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_6_2, %tile_6_3, %tile_6_4, %tile_6_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
    aie.objectfifo @in0_L2L1_7(%mem_tile_7_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {%tile_7_2, %tile_7_3, %tile_7_4, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>
        
    aie.objectfifo.link [@in0_L3L2_0] -> [@in0_L2L1_0]([] [])
    aie.objectfifo.link [@in0_L3L2_1] -> [@in0_L2L1_1]([] [])
    aie.objectfifo.link [@in0_L3L2_2] -> [@in0_L2L1_2]([] [])
    aie.objectfifo.link [@in0_L3L2_3] -> [@in0_L2L1_3]([] [])
    aie.objectfifo.link [@in0_L3L2_4] -> [@in0_L2L1_4]([] [])
    aie.objectfifo.link [@in0_L3L2_5] -> [@in0_L2L1_5]([] [])
    aie.objectfifo.link [@in0_L3L2_6] -> [@in0_L2L1_6]([] [])
    aie.objectfifo.link [@in0_L3L2_7] -> [@in0_L2L1_7]([] [])


    aie.objectfifo @in1_L3L2_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L3L2_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L3L2_2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    aie.objectfifo @in1_L3L2_3(%shim_noc_tile_3_0, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<64x16xbf16>>
    
    aie.objectfifo @in1_L2L1_0(%mem_tile_0_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2, %tile_7_2}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo @in1_L2L1_1(%mem_tile_1_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_3, %tile_1_3, %tile_2_3, %tile_3_3, %tile_4_3, %tile_5_3, %tile_6_3, %tile_7_3}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo @in1_L2L1_2(%mem_tile_2_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_4, %tile_1_4, %tile_2_4, %tile_3_4, %tile_4_4, %tile_5_4, %tile_6_4, %tile_7_4}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    aie.objectfifo @in1_L2L1_3(%mem_tile_3_1 dimensionsToStream [<size = 8, stride = 128>, <size = 2, stride = 8>, <size = 8, stride = 16>, <size = 8, stride = 1>], {%tile_0_5, %tile_1_5, %tile_2_5, %tile_3_5, %tile_4_5, %tile_5_5, %tile_6_5, %tile_7_5}, 2 : i32) : !aie.objectfifo<memref<8x2x8x8xbf16>>
    
    aie.objectfifo.link [@in1_L3L2_0] -> [@in1_L2L1_0]([] [])
    aie.objectfifo.link [@in1_L3L2_1] -> [@in1_L2L1_1]([] [])
    aie.objectfifo.link [@in1_L3L2_2] -> [@in1_L2L1_2]([] [])
    aie.objectfifo.link [@in1_L3L2_3] -> [@in1_L2L1_3]([] [])


    aie.objectfifo @out_L1L2_0_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_1(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_2(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_0_3(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_0(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_1(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_2(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_1_3(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_0(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_1(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_2(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_2_3(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_0(%tile_3_2, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_1(%tile_3_3, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_2(%tile_3_4, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_3_3(%tile_3_5, {%mem_tile_3_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_0(%tile_4_2, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_1(%tile_4_3, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_2(%tile_4_4, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_4_3(%tile_4_5, {%mem_tile_4_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_0(%tile_5_2, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_1(%tile_5_3, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_2(%tile_5_4, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_5_3(%tile_5_5, {%mem_tile_5_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_0(%tile_6_2, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_1(%tile_6_3, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_2(%tile_6_4, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_6_3(%tile_6_5, {%mem_tile_6_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_0(%tile_7_2, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_1(%tile_7_3, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_2(%tile_7_4, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L1L2_7_3(%tile_7_5, {%mem_tile_7_1}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>
    aie.objectfifo @out_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<4x16x16xbf16>>

    aie.objectfifo.link [@out_L1L2_0_0, @out_L1L2_0_1, @out_L1L2_0_2, @out_L1L2_0_3] -> [@out_L2L3_0]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_1_0, @out_L1L2_1_1, @out_L1L2_1_2, @out_L1L2_1_3] -> [@out_L2L3_1]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_2_0, @out_L1L2_2_1, @out_L1L2_2_2, @out_L1L2_2_3] -> [@out_L2L3_2]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_3_0, @out_L1L2_3_1, @out_L1L2_3_2, @out_L1L2_3_3] -> [@out_L2L3_3]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_4_0, @out_L1L2_4_1, @out_L1L2_4_2, @out_L1L2_4_3] -> [@out_L2L3_4]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_5_0, @out_L1L2_5_1, @out_L1L2_5_2, @out_L1L2_5_3] -> [@out_L2L3_5]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_6_0, @out_L1L2_6_1, @out_L1L2_6_2, @out_L1L2_6_3] -> [@out_L2L3_6]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_7_0, @out_L1L2_7_1, @out_L1L2_7_2, @out_L1L2_7_3] -> [@out_L2L3_7]([0, 256, 512, 768] [])



    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_0_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_0(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_0_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_1_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_1(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_1_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_2_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_2(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_2_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_3_2 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_3_5 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_3_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_3(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_3_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_4_2 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_4_3 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_4_4 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_4_5 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_4_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_4(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_4(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_4_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_5_2 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_5_3 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_5_4 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_5_5 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_5_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_5(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_5(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_5_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_6_2 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_6_3 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_6_4 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_6_5 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_6_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_6(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_6(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_6_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_7_2 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_0(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_0(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_0(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_7_3 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_1(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_1(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_1(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_7_4 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_2(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_2(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_2(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}


    %core_7_5 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {
          %buffer_out = aie.objectfifo.acquire @out_L1L2_7_3(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_7(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_7(Consume, 1)
            aie.objectfifo.release @in1_L2L1_3(Consume, 1)
          }
          aie.objectfifo.release @out_L1L2_7_3(Produce, 1)
        }
      }
      aie.end
    } {stack_size = 1024 : i32}

    aie.runtime_sequence(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x128xbf16>, %arg2: memref<256x128xbf16>) {


      
      // a=0
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 16384][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 32768][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 49152][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 65536][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 81920][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 98304][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 114688][2, 16, 16, 64][0, 64, 1024, 1]) {id = 1 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][2, 16, 64, 16][64, 8192, 128, 1]) {id =  3 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 16][2, 16, 64, 16][64, 8192, 128, 1]) {id = 3 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][2, 16, 64, 16][64, 8192, 128, 1]) {id = 3 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 48][2, 16, 64, 16][64, 8192, 128, 1]) {id = 3 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

      // a=1 -> 131072 offset for arg0

      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 131072][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_0} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 147456][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_1} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 163840][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_2} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 180224][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_3} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 196608][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_4} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 212992][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_5} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 229376][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_6} : memref<256x1024xbf16>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 245760][2, 16, 16, 64][0, 64, 1024, 1]) {id = 2 : i64, metadata = @in0_L3L2_7} : memref<256x1024xbf16>
      
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][2, 16, 64, 16][64, 8192, 128, 1]) {id =  4 : i64, metadata = @in1_L3L2_0} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 16][2, 16, 64, 16][64, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_1} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 32][2, 16, 64, 16][64, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_2} : memref<1024x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 48][2, 16, 64, 16][64, 8192, 128, 1]) {id = 4 : i64, metadata = @in1_L3L2_3} : memref<1024x128xbf16>

//a=0, with x offsets per column
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 2048][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 4096][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 6144][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 8192][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 10240][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 12288][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 14336][2, 4, 16, 16][64, 16, 128, 1]) {id = 5 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>
//a=1
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 16384][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_0} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 18432][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_1} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 20480][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_2} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 22528][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_3} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 24576][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_4} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 26624][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_5} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 28672][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_6} : memref<256x128xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 30720][2, 4, 16, 16][64, 16, 128, 1]) {id = 6 : i64, metadata = @out_L2L3_7} : memref<256x128xbf16>

      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_1}
      aiex.npu.dma_wait {symbol = @out_L2L3_2}
      aiex.npu.dma_wait {symbol = @out_L2L3_3}
      aiex.npu.dma_wait {symbol = @out_L2L3_4}
      aiex.npu.dma_wait {symbol = @out_L2L3_5}
      aiex.npu.dma_wait {symbol = @out_L2L3_6}
      aiex.npu.dma_wait {symbol = @out_L2L3_7}
      aiex.npu.dma_wait {symbol = @out_L2L3_0}
      aiex.npu.dma_wait {symbol = @out_L2L3_1}
      aiex.npu.dma_wait {symbol = @out_L2L3_2}
      aiex.npu.dma_wait {symbol = @out_L2L3_3}
      aiex.npu.dma_wait {symbol = @out_L2L3_4}
      aiex.npu.dma_wait {symbol = @out_L2L3_5}
      aiex.npu.dma_wait {symbol = @out_L2L3_6}
      aiex.npu.dma_wait {symbol = @out_L2L3_7}
    }
  }
}

