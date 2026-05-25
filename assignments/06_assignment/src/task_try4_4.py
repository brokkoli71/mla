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
def contraction(A, B, C, m2: ct.Constant[int], n1: ct.Constant[int], l: ct.Constant[int], k: ct.Constant[int], m0: ct.Constant[int], n0: ct.Constant[int]):
    m_temp = ct.bid(0)
    m3_i = m_temp // m2
    m2_i = m_temp % m2

    n2_i = ct.bid(1)

    n_temp = ct.bid(2)
    m1_i = n_temp // n1
    n1_i = n_temp % n1

    acc = ct.zeros((m0, n0), dtype=ct.float32)
    
    for l_i in range(l):
        A_ = ct.load(
            A, 
            index=(m3_i, m2_i, l_i, 0, m1_i, 0), 
            shape=(1,1,1,k,1,m0), 
            padding_mode=ct.PaddingMode.ZERO
        )
        A_ = ct.reshape(A_, (k, m0))
        A_ = ct.transpose(A_)
        B_ = ct.load(
            B, 
            index=(n2_i, l_i, 0, n1_i, 0), 
            shape=(1,1,k,1,n0), 
            padding_mode=ct.PaddingMode.ZERO
        )
        B_ = ct.reshape(B_, (k, n0))
        acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (1, 1, 1, 1, n0, 1, m0))
    ct.store(C, index=(m3_i, n2_i, m2_i, n1_i, 0, m1_i, 0), tile=acc)



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
    
    tensor_torch_acspx_16 = torch.tensor(tensor_acspx_16)
    tensor_torch_bspy_16 = torch.tensor(tensor_bspy_16)
    

    config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    file_dir = Path(__file__).parent


    #opti = Optimizer(config)
    #print(opti.config)
    print(tensor_acspx_16.shape, tensor_bspy_16.shape)
    #opti.split_dim(6, outer_size=None, inner_size=128)
    #opti.split_dim(4, outer_size=None, inner_size=128)
    #print(opti.config)

    tensor_acspx_16 = tensor_acspx_16.unflatten(dim=4, sizes=(-1,128))
    tensor_bspy_16 = tensor_bspy_16.unflatten(dim=3, sizes=(-1,128))
    print(tensor_acspx_16.shape, tensor_bspy_16.shape)



    # 3. Prepare Tensor C
    # Layout: (a, b, c, n1, n0, m1, m0)
    # abcyx
    C_m2m1x_n2n1y = torch.empty((4, 4, 3, 9, 128, 12, 128), dtype=torch.float16, device='cuda')


    grid = (4*3, 4, 12*9)

    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (tensor_acspx_16, tensor_bspy_16, C_m2m1x_n2n1y,  3, 9, 64, 64 ,128, 128)
    )

    C_final = C_m2m1x_n2n1y.flatten(5,6).flatten(3,4)
    print(C_final.shape)


    expected = torch.einsum(einsum_string, tensor_torch_acspx_16, tensor_torch_bspy_16)
    assert torch.allclose(C_final, expected, atol=2e-0), "The result is incorrect!"
    print("The result is correct!")

    plot_tensor(
        C_final.to('cpu'),
        path=file_dir / 'results' / 'try_4_4_torch_16.png',
        title='Lightfield Tensorring Decomposition - PyTorch (Float16)'
    )
    
    # ----------------------------------------------------------------
    # Benchmark torch.einsum
    # ----------------------------------------------------------------
    t_ms_torch = triton.testing.do_bench(lambda: torch.einsum(einsum_string, tensor_torch_acspx_16, tensor_torch_bspy_16))   
    
    # Dimensionen auslesen für korrekte FLOP-Berechnung
    a, c, s, p, x = tensor_torch_acspx_16.shape
    b, _, _, y = tensor_torch_bspy_16.shape
    
    # Korrekte FLOP-Formel: 2 * (Produkt aller relevanten Dimensionen)
    flops = 2 * (a * b * c * s * p * x * y)
    
    tflops_torch = flops / (t_ms_torch / 1000) / (10**12)
    
    print(f"torch.einsum:")
    print(f"Execution time of torch einsum: {t_ms_torch:.2f} ms")
    print(f"TFLOPS of torch einsum: {tflops_torch:.2f}")

    # ----------------------------------------------------------------
    # Benchmark optimized kernel
    # ----------------------------------------------------------------
    t_ms_opt = triton.testing.do_bench(lambda:
        ct.launch(
            torch.cuda.current_stream(), 
            grid, 
            contraction, 
            (tensor_acspx_16, tensor_bspy_16, C_m2m1x_n2n1y,  3, 9, 64, 64 ,128, 128)
    )
    )
    
    tflops_opt = flops / (t_ms_opt / 1000) / (10**12)
    
    print(f"\nOptimized kernel:")
    print(f"Execution time of optimized kernel: {t_ms_opt:.2f} ms")
    print(f"TFLOPS of optimized kernel: {tflops_opt:.2f}")

