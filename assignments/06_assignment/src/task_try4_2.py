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

#'acKx,bKy->abcyx'
# torch.Size([12, 12, 128, 4096]) torch.Size([3, 12, 4096, 128])
@ct.kernel
def contraction(A, B, C, n1: ct.Constant[int], k: ct.Constant[int], x: ct.Constant[int], y: ct.Constant[int]):
    
    m2_i = ct.bid(2) 
    n2_i = ct.bid(1)  

    swizzle = ct.bid(0)     
    m1_i = swizzle % n1 
    n1_i = swizzle // n1    

    #    grid = (6*6*6*24,1,1)

    # temp = ct.bid(0)

    # n1_i = temp % 6
    # temp = temp // 6

    # m1_i = temp % 6
    # temp = temp // 6

    # n2_i = temp % 6
    # m2_i = temp // 24



    acc = ct.zeros((x,y), dtype=ct.float16)
    
    k_t = 64

    for k_i in range(64):
        A_ = ct.load(
            A, 
            index=(m2_i, m1_i, 0, k_i), 
            shape=(1,1,x,k_t),
            padding_mode=ct.PaddingMode.ZERO
        )
        A_ = ct.reshape(A_, (x, k_t))
        #A_ = ct.transpose(A_)
        B_ = ct.load(
            B, 
            index=(n2_i, n1_i, k_i, 0), 
            shape=(1,1,k_t,y), 
            padding_mode=ct.PaddingMode.ZERO
        )
        B_ = ct.reshape(B_, (k_t, y))
        #B_ = ct.transpose(B_)
        acc += ct.matmul(A_, B_)

    #acc = ct.astype(acc, ct.float16)

    acc = ct.reshape(acc, (1, 1, x, 1, 1, y))
    ct.store(C, index=(m2_i, m1_i, 0, n2_i, n1_i, 0), tile=acc)



if __name__ == "__main__":

    file_dir = Path(__file__).parent
    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    data = np.load(file_dir / '../data' / 'lf_tr_64_intermediate.npz')
    tensor_acspx = torch.tensor(data['tensor_acspx'])
    tensor_bspy = torch.tensor(data['tensor_bspy'])

    # Convert all tensors to torch tensors and move them to the GPU before calling `torch.einsum`. Run the contraction **twice**: once with `torch.float32` inputs and once with `torch.float16` inputs (cast the tensors before contracting).
    einsum_string = 'acspx,bspy->abcyx'
    # M = acx  N = by  K = sp   C =

    a = 4
    c = 3
    s = 64
    p = 64
    x = 1536
    b = 4
    y = 1152

    tensor_acspx_32 = tensor_acspx.to('cuda')
    tensor_bspy_32 = tensor_bspy.to('cuda')
    
    tensor_acspx_16 = tensor_acspx.to('cuda').to(torch.float16)
    tensor_bspy_16 = tensor_bspy.to('cuda').to(torch.float16)

    print(tensor_acspx_16.shape, tensor_bspy_16.shape)

    # A war (a, c, s, p, x) -> wird (ac, K, x)
    tensor_acKx = tensor_acspx_16.flatten(2, 3).flatten(0, 1).permute(0, 2, 1)

    
    # B war (b, s, p, y) -> wird (b, K, y)
    tensor_bKy = tensor_bspy_16.flatten(1, 2)

    K = s * p

    print(tensor_acKx.shape, tensor_bKy.shape)
    tensor_acKx = tensor_acKx.unflatten(dim=1, sizes=( 12, 128)).flatten(0,1)
    tensor_bKy = tensor_bKy.unflatten(dim=2, sizes=( 9, 128)).permute(0, 2, 1, 3).flatten(0,1)
    print(tensor_acKx.shape, tensor_bKy.shape)
    tensor_acKx = tensor_acKx.unflatten(dim=0, sizes=(24,6)).contiguous()
    tensor_bKy = tensor_bKy.unflatten(dim=0, sizes=(6,6)).contiguous()
    print(tensor_acKx.shape, tensor_bKy.shape)




    #config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    #print(config)

    # 3. Prepare Tensor C
    # Layout: (ac, b, m2, m1, x_block, n2, n1, y_block)
    # Größen: (12, 4,  4,  6,       128,  3,  6,       128)
    C_m2m1x_n2n1y = torch.empty((24, 6, 128, 6, 6, 128), dtype=torch.float16, device='cuda')

    #grid = (6*6*6*24,1,1)
    grid = (6*6,6,24)

    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (tensor_acKx, tensor_bKy, C_m2m1x_n2n1y, 6, K, 128, 128)
    )

    C_final = C_m2m1x_n2n1y
    
    #print("Shape original:", C_final.shape)
    
    # Wir verschmelzen nun die y-Dimensionen (n2, n1, y_block) -> Dim 5, 6, 7
    # 3 * 6 * 64 = 1152
    #C_final = C_final.flatten(5, 7)
    #print("Nach y-Flatten:", C_final.shape) # (12, 4, 4, 6, 64, 1152)
    
    # Wir verschmelzen die x-Dimensionen (m2, m1, x_block) -> Dim 2, 3, 4
    # 4 * 6 * 64 = 1536
    #C_final = C_final.flatten(2, 4)
    #print("Nach x-Flatten:", C_final.shape) # (12, 4, 1536, 1152) -> (ac, b, x, y)
    
    # a*c (12) wieder aufteilen in a(4) und c(3) und in die Ziel-Reihenfolge (a, b, c, y, x) bringen
    #C_final = C_final.unflatten(dim=0, sizes=(4, 3)) # (4, 3, 4, 1536, 1152) -> (a, c, b, x, y)
    #C_final = C_final.permute(0, 2, 1, 4, 3).contiguous() # (a, b, c, y, x)
    #print("Finales Shape:", C_final.shape)

    expected = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)
    #assert torch.allclose(C_final, expected, atol=1e-0), "The result is incorrect!"
    #print("The result is correct!")

    # plot_tensor(
    #     C_final.to('cpu'),
    #     path=file_dir / 'results' / 'try4_2_torch_16.png',
    #     title='Lightfield Tensorring Decomposition - PyTorch (Float16)'
    # )
    
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
    torch.cuda.init()
    t_ms_opt = triton.testing.do_bench(lambda:
    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (tensor_acKx, tensor_bKy, C_m2m1x_n2n1y, 6, K, 128, 128)
    )
    )
    torch.cuda.synchronize()
    
    tflops_opt = flops / (t_ms_opt / 1000) / (10**12)
    
    print(f"\nOptimized kernel:")
    print(f"Execution time of optimized kernel: {t_ms_opt:.2f} ms")
    print(f"TFLOPS of optimized kernel: {tflops_opt:.2f}")

