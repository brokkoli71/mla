import itertools


x = 0
y = 0
str = "".join(f"""
    %core_{x}_{y+2} = aie.core(%tile_{x}_{y+2}) {{
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

str += "".join(f"""
    aie.objectfifo @out_L1L2_{x}_{y}(%tile_{x}_{y+2}, {{%mem_tile_{x}_1}}, 2 : i32) : !aie.objectfifo<memref<2x2x8x8xbf16>>"""
    for x,y in itertools.product(range(8), range(4)))

str += "".join(f"""
    aie.objectfifo @out_L2L3_{x}(%mem_tile_{x}_1 dimensionsToStream [<size = 2, stride = 128>, <size = 8, stride = 8>, <size = 2, stride = 64>, <size = 8, stride = 1>], {{%shim_noc_tile_{x}_0}}, 2 : i32) : !aie.objectfifo<memref<16x16xbf16>>"""
    for x in range(8))

str += "".join(f"""
    aie.objectfifo.link [@out_L1L2_{x}_0, @out_L1L2_{x}_1, @out_L1L2_{x}_2, @out_L1L2_{x}_3] -> [@out_L2L3_{x}]([0, 256, 512, 768] [])"""
    for x in range(8))

str += "\n//a=0, with x offsets per column"

str += "".join(f"""
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, {2048*x}][2, 4, 16, 16][64, 16, 128, 1]) {{id = 0 : i64, metadata = @out_L2L3_{x}}} : memref<256x128xbf16>"""
    for x in range(8))
str += "\n//a=1"
str += "".join(f"""
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, {2048*x+16384}][2, 4, 16, 16][64, 16, 128, 1]) {{id = 5 : i64, metadata = @out_L2L3_{x}}} : memref<256x128xbf16>"""
    for x in range(8))




str += "".join(f"""
    aie.objectfifo @in0_L2L1_{x}(%mem_tile_{x}_1 dimensionsToStream [<size = 2, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>], {{%tile_{x}_2, %tile_{x}_3, %tile_{x}_4, %tile_{x}_5}}, 2 : i32) : !aie.objectfifo<memref<2x8x8x8xbf16>>"""
    for x in range(8))

# print to file
with open(__file__.replace(".py", "d.mlir"), "w") as f:
    f.write(str)

