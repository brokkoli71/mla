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
    # grid = (a *c, 4, 3 * 6 * 4 * 6) # m3, n3, n2 * n1 * m2 * m1

    # ct.launch(
    #     torch.cuda.current_stream(), 
    #     grid, 
    #     contraction, 
    #     (tensor_acKx, tensor_bKy, C_m2n2n1n0m1m0, 6, 4, 6 ,K, 64, 64)
    # )
@ct.kernel
def contraction(A, B, C, n1: ct.Constant[int], m2: ct.Constant[int], m1: ct.Constant[int], k: ct.Constant[int], x: ct.Constant[int], y: ct.Constant[int]):
    
    m3_i = ct.bid(0)  # Das ist (a * c) aus grid[0]
    n3_i = ct.bid(1)  # Das ist die (4) aus grid[1]

    # 2. Die verschmolzene Dimension (n2 * n1 * m2 * m1) abgreifen
    bc_it = ct.bid(2) # Das ist (3 * 6 * 4 * 6) aus grid[2]
    
    # 3. Den 1D-Index in 4 Indizes auflösen (von innen nach außen abspalten)
    # Reihenfolge in der Multiplikation: n2 * n1 * m2 * m1 -> m1 ist die innerste Dimension
    m1_i = bc_it % m1
    temp = bc_it // m1
    
    m2_i = temp % m2
    temp = temp // m2
    
    n1_i = temp % n1
    n2_i = temp // n1  
    
    acc = ct.zeros((x,y), dtype=ct.float32)
    
    for k_i in range(0, k, 128):
        A_ = ct.load(
            A, 
            index=(m3_i, m2_i, m1_i, 0, k_i), 
            shape=(1,1,1,x,128),
            padding_mode=ct.PaddingMode.ZERO
        )
        A_ = ct.reshape(A_, (x, 128))
        B_ = ct.load(
            B, 
            index=(n3_i, n2_i, n1_i, k_i, 0), 
            shape=(1,1,1,128,y), 
            padding_mode=ct.PaddingMode.ZERO
        )
        B_ = ct.reshape(B_, (128, y))
        acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    # abcyx (12, 4, 3, 6, 64, 4, 6, 64)
    acc = ct.reshape(acc, (1,1,1,1,y,1,1,x))
    ct.store(C, index=(m3_i, n3_i, n2_i, n1_i, 0, m2_i, m1_i, 0), tile=acc)



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

    # A war (a, c, s, p, x) -> wird (a, c, K, x)
    tensor_acKx = tensor_acspx_16.flatten(2, 3).flatten(0, 1).permute(0, 2, 1)

    
    # B war (b, s, p, y) -> wird (b, K, y)
    tensor_bKy = tensor_bspy_16.flatten(1, 2)

    K = s * p

    print(tensor_acKx.shape, tensor_bKy.shape)
    tensor_acKx = tensor_acKx.unflatten(dim=1, sizes=( 24, 64)).unflatten(dim=1, sizes=( 4, 6)).contiguous()
    tensor_bKy = tensor_bKy.unflatten(dim=2, sizes=( 18, 64)).permute(0, 2, 1, 3).unflatten(dim=1, sizes=( 3, 6)).contiguous()
    print(tensor_acKx.shape, tensor_bKy.shape)



    #config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    #print(config)

    # 3. Prepare Tensor C
    # abcyx
    # Sizes: a=4, c=3, x=1536, b=4, y=1152
    # a*c = 12, n1=b=4, n2=y=1152
    C_m2n2n1n0m1m0 = torch.empty((12, 4, 3, 6, 64, 4, 6, 64), dtype=torch.float16, device='cuda')

    grid = (a *c, 4, 3 * 6 * 4 * 6) # m3, n3, n2 * n1 * m2 * m1

    ct.launch(
        torch.cuda.current_stream(), 
        grid, 
        contraction, 
        (tensor_acKx, tensor_bKy, C_m2n2n1n0m1m0, 6, 4, 6 ,K, 64, 64)
    )

    C_final = C_m2n2n1n0m1m0
    
    # print(C_final.shape)
    # C_final = C_final.flatten(4,5).flatten(2,3)
    # print(C_final.shape)
    # C_final = C_final.unflatten(dim=0, sizes=(4, 3)).permute(0, 2, 1, 3, 4).contiguous()
    # print(C_final.shape)
    
    print(C_final.shape)
    C_final = C_final.flatten(5,6).flatten(2,3)
    print(C_final.shape)
    C_final = C_final.flatten(4,5).flatten(2,3)
    print(C_final.shape)
    C_final = C_final.unflatten(dim=0, sizes=(4, 3)).permute(0, 2, 1, 3, 4).contiguous()
    print(C_final.shape)


    expected = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)
    assert torch.allclose(C_final, expected, atol=2e-0), "The result is incorrect!"
    print("The result is correct!")

    plot_tensor(
        C_final.to('cpu'),
        path=file_dir / 'results' / 'try4_torch_16.png',
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
        (tensor_acKx, tensor_bKy, C_m2n2n1n0m1m0, 6, 4, 6 ,K, 64, 64)
    ))
    
    tflops_opt = flops / (t_ms_opt / 1000) / (10**12)
    
    print(f"\nOptimized kernel:")
    print(f"Execution time of optimized kernel: {t_ms_opt:.2f} ms")
    print(f"TFLOPS of optimized kernel: {tflops_opt:.2f}")

