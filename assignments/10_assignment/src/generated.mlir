
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_0_2 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_0_2 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_0_2 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_1_2 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_1_2 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_1_2 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_2_2 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_2_2 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_2_2 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_3_2 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_3_2 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_3_2 = aie.core(%tile_3_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_4_2 = aie.core(%tile_4_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_4_2 = aie.core(%tile_4_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_4_2 = aie.core(%tile_4_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_5_2 = aie.core(%tile_5_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_5_2 = aie.core(%tile_5_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_5_2 = aie.core(%tile_5_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_6_2 = aie.core(%tile_6_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_6_2 = aie.core(%tile_6_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_6_2 = aie.core(%tile_6_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
        %c4 = arith.constant 128 : index
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


    %core_7_2 = aie.core(%tile_7_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_7_2 = aie.core(%tile_7_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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


    %core_7_2 = aie.core(%tile_7_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
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
    aie.objectfifo @out_L2L3_0(%mem_tile_0_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_1(%mem_tile_1_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_2(%mem_tile_2_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_3(%mem_tile_3_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_3_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_4(%mem_tile_4_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_4_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_5(%mem_tile_5_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_5_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_6(%mem_tile_6_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_6_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo @out_L2L3_7(%mem_tile_7_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {%shim_noc_tile_7_0}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>
    aie.objectfifo.link [@out_L1L2_0_0, out_L1L2_0_1, out_L1L2_0_2, out_L1L2_0_3] -> [@out_L2L3_0]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_1_0, out_L1L2_1_1, out_L1L2_1_2, out_L1L2_1_3] -> [@out_L2L3_1]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_2_0, out_L1L2_2_1, out_L1L2_2_2, out_L1L2_2_3] -> [@out_L2L3_2]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_3_0, out_L1L2_3_1, out_L1L2_3_2, out_L1L2_3_3] -> [@out_L2L3_3]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_4_0, out_L1L2_4_1, out_L1L2_4_2, out_L1L2_4_3] -> [@out_L2L3_4]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_5_0, out_L1L2_5_1, out_L1L2_5_2, out_L1L2_5_3] -> [@out_L2L3_5]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_6_0, out_L1L2_6_1, out_L1L2_6_2, out_L1L2_6_3] -> [@out_L2L3_6]([0, 256, 512, 768] [])
    aie.objectfifo.link [@out_L1L2_7_0, out_L1L2_7_1, out_L1L2_7_2, out_L1L2_7_3] -> [@out_L2L3_7]([0, 256, 512, 768] [])