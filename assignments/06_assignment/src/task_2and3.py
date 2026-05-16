import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import triton
import cupy as cp
import cuda.tile as ct

file_dir = Path(__file__).parent
assignment_05_src = (file_dir / '../../05_assignment/src').resolve()
# Add it to Python's search path
sys.path.append(str(assignment_05_src))

from optimizer import Optimizer
from config import Config, DataType, PrimType, DimType, ExecType, generate_config





if __name__ == "__main__":

    file_dir = Path(__file__).parent
    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    data = np.load(file_dir / '../data' / 'lf_tr_64_intermediate.npz')
    tensor_acspx = torch.tensor(data['tensor_acspx'])
    tensor_bspy = torch.tensor(data['tensor_bspy'])

    # Convert all tensors to torch tensors and move them to the GPU before calling `torch.einsum`. Run the contraction **twice**: once with `torch.float32` inputs and once with `torch.float16` inputs (cast the tensors before contracting).
    einsum_string = 'acspx,bspy->abcyx'

    tensor_acspx_32 = tensor_acspx.to('cuda')
    tensor_bspy_32 = tensor_bspy.to('cuda')
    
    tensor_acspx_16 = tensor_acspx.to('cuda').to(torch.float16)
    tensor_bspy_16 = tensor_bspy.to('cuda').to(torch.float16)
    #C = torch.empty(tensor_acspx_16.shape[0], tensor_bspy_16.shape[0], tensor_acspx_16.shape[2], tensor_bspy_16.shape[2], device='cuda', dtype=torch.float16)

    config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    file_dir = Path(__file__).parent

    #print(config);
    # with open(file_dir / "results" / "task2_config.out", "w") as f:
    #     f.write(str(config))


    ## first try maximixe tile size 128y128x128
    # opti = Optimizer(config)
    # opti.fuse_dims(2,3)
    # opti.split_dim(2, outer_size=None, inner_size=128)
    # #print(opti.config)
    # print(opti.make_executable())
    # opti.split_dim(4 , outer_size=None, inner_size=128)
    # print(opti.make_executable())
    # opti.split_dim(6 , outer_size=None, inner_size=128)
    # print(opti.make_executable())

    opti = Optimizer(config)
    print(opti.make_executable())
    opti.split_dim(4, outer_size=None, inner_size=64)
    print(opti.make_executable())
    opti.split_dim(6, outer_size=None, inner_size=32)
    print(opti.make_executable())

    opt_config = opti.make_executable()


    # 1. Prepare Tensor A
    # tensor_acspx_16 is already in the correct physical memory layout.
    # We just map it to the 9D layout required by the opt_config.
    A_opt = torch.as_strided(
        tensor_acspx_16, 
        size=opt_config.dim_sizes, 
        stride=opt_config.strides[0]
    )

    # 2. Prepare Tensor B
    # The config expects contiguous layout (s, p, b, y).
    # Original bspy is dims (0, 1, 2, 3), so we permute to (1, 2, 0, 3) and make contiguous
    tensor_spby = tensor_bspy_16.permute(1, 2, 0, 3).contiguous()
    B_opt = torch.as_strided(
        tensor_spby, 
        size=opt_config.dim_sizes, 
        stride=opt_config.strides[1]
    )

    # 3. Prepare Tensor C
    # The config expects contiguous layout (a, c, x, b, y). 
    # Sizes: a=4, c=3, x=1536, b=4, y=1152
    C_acxby = torch.empty((4, 3, 1536, 4, 1152), dtype=torch.float16, device='cuda')
    C_opt = torch.as_strided(
        C_acxby, 
        size=opt_config.dim_sizes, 
        stride=opt_config.strides[2]
    )

    grid = (opt_config.dim_sizes[0], opt_config.dim_sizes[1], opt_config.dim_sizes[2] * opt_config.dim_sizes[3] * opt_config.dim_sizes[4])
    

    #ct.launch(torch.cuda.current_stream(), grid, contraction, (tensor_acspx_16, tensor_bspy_16, opt_config.dim_sizes[5], opt_config.dim_sizes[6], opt_config.dim_sizes[7], opt_config.dim_sizes[8]))
    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (A_opt, B_opt, C_opt, opt_config.dim_sizes[5], opt_config.dim_sizes[6], opt_config.dim_sizes[7], opt_config.dim_sizes[8])
    )

    expected = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)
    assert torch.allclose(C, expected, atol=1e-0), "The result is incorrect!"

    
    
    #Benchmark the kernel
    t_ms = triton.testing.do_bench(lambda: torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16))   
    tflops = 2 * (tensor_acspx_16.shape[0] * tensor_bspy_16.shape[0] * tensor_acspx_16.shape[2] * tensor_bspy_16.shape[2]) / (t_ms / 1000) / (10**12)
    print(f"torch.einsum:")
    print(f"Time: {t_ms:.2f} ms")
    print(f"Execution time of torch einsum: {t_ms:.2f} ms")
    print(f"TFLOPS of torch einsum: {tflops:.2f}")

    
    # t_ms = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, contraction, (tensor_acspx_16, tensor_bspy_16, C, k, l, x_padded, y_padded, z_padded, c)))    
    # print(f"Optimized kernel:")
    # print(f"Time: {t_ms:.2f} ms")
    #tflops = 2 * (tensor_acspx_16.shape[0] * tensor_bspy_16.shape[0] * tensor_acspx_16.shape[2] * tensor_bspy_16.shape[2]) / (t_ms / 1000) / (10**12)
    # print(f"Execution time of optimized kernel: {t_ms:.2f} ms")
    # print(f"TFLOPS of optimized kernel: {tflops:.2f}")



@ct.kernel
def contraction(A, B, C, l: ct.Constant[int], m: ct.Constant[int], n: ct.Constant[int], k: ct.Constant[int]):
    e_it = ct.bid(0)
    a_it = ct.bid(1)
    bc_it = ct.bid(2)
    b_it = bc_it // c
    c_it = bc_it % c

    acc = ct.zeros((x, z), dtype=ct.float32)
    
    for k_it in range(k):
        for l_it in range(l):
            A_ = ct.load(
                A, 
                index=(e_it,a_it,b_it,k_it,l_it,0,0), 
                shape=(1,1,1,1,1,x,y), 
                padding_mode=ct.PaddingMode.ZERO
            )
            A_ = ct.reshape(A_, (x, y))
            B_ = ct.load(
                B, 
                index=(e_it,c_it,k_it,l_it,0,0), 
                shape=(1,1,1,1,y,z), 
                padding_mode=ct.PaddingMode.ZERO
            )
            B_ = ct.reshape(B_, (y, z))
            acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (1,1,1,1,x,z))
    ct.store(C, index=(e_it,a_it,b_it,c_it,0,0), tile=acc)