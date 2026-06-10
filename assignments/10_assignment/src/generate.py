import itertools


x = 0
y = 0
str = "".join(f"""
    %core_{x}_2 = aie.core(%tile_{x}_{y+2}) {{
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {{
        %c0_1 = arith.constant 0 : index
        %c4 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %i_ab = %c0_1 to %c4 step %c1_1 {{
          %buffer_out = aie.objectfifo.acquire @out_L1L2_{x}_{y}(Produce, 1) : !aie.objectfifosubview<memref<2x2x8x8xbf16>>
          %out = aie.objectfifo.subview.access %buffer_out[0] : !aie.objectfifosubview<memref<2x2x8x8xbf16>> -> memref<2x2x8x8xbf16>
          func.call @zero(%out) : (memref<2x2x8x8xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c16 = arith.constant 16 : index
          %c1_2 = arith.constant 1 : index
          scf.for %i_c = %c0_2 to %c16 step %c1_2 {{
            %buffer_in0 = aie.objectfifo.acquire @in0_L2L1_{x}(Consume, 1) : !aie.objectfifosubview<memref<2x8x8x8xbf16>>
            %in0 = aie.objectfifo.subview.access %buffer_in0[0] : !aie.objectfifosubview<memref<2x8x8x8xbf16>> -> memref<2x8x8x8xbf16>
            %buffer_in1 = aie.objectfifo.acquire @in1_L2L1_{y}(Consume, 1) : !aie.objectfifosubview<memref<8x2x8x8xbf16>>
            %in1 = aie.objectfifo.subview.access %buffer_in1[0] : !aie.objectfifosubview<memref<8x2x8x8xbf16>> -> memref<8x2x8x8xbf16>
            func.call @matmul(%in0, %in1, %out) : (memref<2x8x8x8xbf16>, memref<8x2x8x8xbf16>, memref<2x2x8x8xbf16>) -> ()
            aie.objectfifo.release @in0_L2L1_{x}(Consume, 1)
            aie.objectfifo.release @in1_L2L1_{y}(Consume, 1)
          }}
          aie.objectfifo.release @out_L1L2_{x}_{y}(Produce, 1)
        }}
      }}
      aie.end
    }} {{stack_size = 1024 : i32}}

""" for x,y in itertools.product(range(8), range(4)))

# print to file
with open(__file__.replace(".py", "d.mlir"), "w") as f:
    f.write(str)