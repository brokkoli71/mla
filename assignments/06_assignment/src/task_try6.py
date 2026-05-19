import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import triton
import cupy as cp
import cuda.tile as ct
from task_1 import plot_tensor

file_dir = Path(__file__).parent
assignment_05_src = (file_dir / '../../05_assignment/src').resolve()
# Add it to Python's search path
sys.path.append(str(assignment_05_src))

from optimizer import Optimizer
from config import Config, DataType, PrimType, DimType, ExecType, generate_config

@ct.kernel
def contraction(A, B, C, 
                m5: ct.Constant[int], m4: ct.Constant[int], 
                n3: ct.Constant[int], m2: ct.Constant[int], n2: ct.Constant[int], 
                m1: ct.Constant[int], n1: ct.Constant[int], 
                m0: ct.Constant[int], k: ct.Constant[int], n0: ct.Constant[int]):

    # 1. Block ID 0: m5 * m4
    bid_0 = ct.bid(0)
    m4_i = bid_0 % m4
    m5_i = bid_0 // m4

    # 2. Block ID 1: n3 * m2 * n2
    bid_1 = ct.bid(1)
    n2_i = bid_1 % n2
    temp1 = bid_1 // n2
    m2_i = temp1 % m2
    n3_i = temp1 // m2

    # 3. Block ID 2: m1 * n1
    bid_2 = ct.bid(2)
    n1_i = bid_2 % n1
    m1_i = bid_2 // n1

    acc = ct.zeros((m0, n0), dtype=ct.float32)

    k_t = 64
    
    for k_i in range(0, k, k_t):
        A_ = ct.load(
            A, 
            index=(m5_i, m4_i, n3_i, m2_i, n2_i, m1_i, n1_i, 0, k_i, 0), 
            shape=(1, 1, 1, 1, 1, 1, 1, m0, k_t, 1), 
            padding_mode=ct.PaddingMode.ZERO
        )
        A_ = ct.reshape(A_, (m0, k_t))
        B_ = ct.load(
            B, 
            index=(m5_i, m4_i, n3_i, m2_i, n2_i, m1_i, n1_i, 0, k_i, 0), 
            shape=(1, 1, 1, 1, 1, 1, 1, 1, k_t, n0), 
            padding_mode=ct.PaddingMode.ZERO
        )
        B_ = ct.reshape(B_, (k_t, n0))
        acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (1, 1, 1, 1, 1, 1, 1, m0, 1, n0))
    ct.store(C, index=(m5_i, m4_i, n3_i, m2_i, n2_i, m1_i, n1_i, 0, 0, 0), tile=acc)



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

    config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    file_dir = Path(__file__).parent


    opti = Optimizer(config)
    print(opti.config)
    opti.fuse_dims(2,3)
    print(opti.make_executable())
    opti.split_dim(4, outer_size=None, inner_size=64)
    print(opti.make_executable())
    opti.split_dim(4, outer_size=None, inner_size=64)
    opti.make_executable()
    opti.split_dim(4, outer_size=None, inner_size=6)
    opti.split_dim(3, outer_size=None, inner_size=6)
    opti.make_executable()
    opti.permute_dims([0, 1, 2, 3, 5, 4, 6, 7, 9, 8])
    
    print(opti.config)


    # 1. Prepare Tensor A
    # tensor_acspx_16 is already in the correct physical memory layout.
    # We just map it to the 9D layout required by the opti.config.
    A_opt = torch.as_strided(
        tensor_acspx_16, 
        size=opti.config.dim_sizes, 
        stride=opti.config.strides[0]
    )

    # 2. Prepare Tensor B
    # The config expects contiguous layout (s, p, b, y).
    tensor_spby = tensor_bspy_16.permute(1, 2, 0, 3).contiguous()
    B_opt = torch.as_strided(
        tensor_spby, 
        size=opti.config.dim_sizes, 
        stride=opti.config.strides[1]
    )

    # 3. Prepare Tensor C
    # The config expects contiguous layout (a, c, x, b, y). 
    # Sizes: a=4, c=3, x=1536, b=4, y=1152
    C_acxby = torch.empty((4, 3, 1536, 4, 1152), dtype=torch.float16, device='cuda')
    C_opt = torch.as_strided(
        C_acxby, 
        size=opti.config.dim_sizes, 
        stride=opti.config.strides[2]
    )

    grid = (
        opti.config.dim_sizes[0] * opti.config.dim_sizes[1],                           # bid(0): m5 und m4
        opti.config.dim_sizes[2] * opti.config.dim_sizes[3] * opti.config.dim_sizes[4], # bid(1): n3, m2, n2
        opti.config.dim_sizes[5] * opti.config.dim_sizes[6]                            # bid(2): m1, n1
    )

    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (A_opt, B_opt, C_opt, 
         opti.config.dim_sizes[0], opti.config.dim_sizes[1], opti.config.dim_sizes[2], 
         opti.config.dim_sizes[3], opti.config.dim_sizes[4], opti.config.dim_sizes[5], 
         opti.config.dim_sizes[6], opti.config.dim_sizes[7], opti.config.dim_sizes[8], 
         opti.config.dim_sizes[9])
    )

    C_final = C_acxby.permute(0, 3, 1, 4, 2)

    expected = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)
    assert torch.allclose(C_final, expected, atol=2e-0), "The result is incorrect!"
    print("The result is correct!")

    plot_tensor(
        C_final.to('cpu'),
        path=file_dir / 'results' / 'task_try6_torch_16.png',
        title='Lightfield Tensorring Decomposition - PyTorch (Float16)'
    )
    
    # ----------------------------------------------------------------
    # Benchmark torch.einsum
    # ----------------------------------------------------------------
    t_ms_torch = triton.testing.do_bench(lambda: torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16))   
    
    # Dimensionen auslesen für korrekte FLOP-Berechnung
    a, c, s, p, x = tensor_acspx_16.shape
    b, _, _, y = tensor_bspy_16.shape
    
    # Korrekte FLOP-Formel: 2 * (Produkt aller relevanten Dimensionen)
    flops = 2 * (a * b * c * s * p * x * y)
    
    tflops_torch = flops / (t_ms_torch / 1000) / (10**12)
    
    print(f"torch.einsum:")
    print(f"Execution time of torch einsum: {t_ms_torch:.2f} ms")
    print(f"TFLOPS of torch einsum: {tflops_torch:.2f}")

    # ----------------------------------------------------------------
    # Benchmark optimized kernel
    # ----------------------------------------------------------------
    t_ms_opt = triton.testing.do_bench(lambda: ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (A_opt, B_opt, C_opt, 
         opti.config.dim_sizes[0], opti.config.dim_sizes[1], opti.config.dim_sizes[2], 
         opti.config.dim_sizes[3], opti.config.dim_sizes[4], opti.config.dim_sizes[5], 
         opti.config.dim_sizes[6], opti.config.dim_sizes[7], opti.config.dim_sizes[8], 
         opti.config.dim_sizes[9])
    ))
    
    tflops_opt = flops / (t_ms_opt / 1000) / (10**12)
    
    print(f"\nOptimized kernel:")
    print(f"Execution time of optimized kernel: {t_ms_opt:.2f} ms")
    print(f"TFLOPS of optimized kernel: {tflops_opt:.2f}")

