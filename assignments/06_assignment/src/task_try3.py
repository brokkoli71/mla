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

# Config nach den beiden split_dim-Aufrufen (9 Dimensionen!):
# Config(
#     data_type=DataType.FLOAT16,
#     prim_main=PrimType.GEMM,
#     prim_last=LastType.NONE,
#     prim_first=FirstType.ZERO,
#     dim_types=[<DimType.M: 0>, <DimType.M: 0>, <DimType.N: 1>, <DimType.M: 0>, <DimType.N: 1>, <DimType.K: 2>, <DimType.M: 0>, <DimType.N: 1>, <DimType.K: 2>],
#     exec_types=[<ExecType.PAR: 1>, <ExecType.PAR: 1>, <ExecType.PAR: 1>, <ExecType.PAR: 1>, <ExecType.PAR: 1>, <ExecType.SEQ: 0>, <ExecType.PRIM: 2>, <ExecType.PRIM: 2>, <ExecType.PRIM: 2>],
#     dim_sizes=[4, 3, 4, 12, 9, 64, 128, 128, 64],
#     strides=[[18874368, 6291456, 0, 128, 0, 98304, 1, 0, 1536], [0, 0, 4718592, 0, 128, 73728, 0, 1, 1152], [21233664, 1769472, 5308416, 128, 196608, 0, 1, 1536, 0]]
# )
#
# Bedeutung der Dimensionen:
#   0: a  = 4    (M, PAR)      3: x2 = 12   (M, PAR)      6: x1 = 128 (M, PRIM)
#   1: c  = 3    (M, PAR)      4: y2 = 9    (N, PAR)      7: y1 = 128 (N, PRIM)
#   2: b  = 4    (N, PAR)      5: s  = 64   (K, SEQ)      8: p  = 64  (K, PRIM)
@ct.kernel
def contraction(A, B, C, m1: ct.Constant[int], n1: ct.Constant[int], k: ct.Constant[int], m0: ct.Constant[int], n0: ct.Constant[int], k0: ct.Constant[int]):
    m3_i = ct.bid(0)    # dim 0 -> a
    m2_i = ct.bid(1)    # dim 1 -> c

    bc_it = ct.bid(2)
    n1_i = bc_it % n1   # dim 4 -> y2 (n1 = 9)
    temp = bc_it // n1
    m1_i = temp % m1    # dim 3 -> x2 (m1 = 12)
    n2_i = temp // m1   # dim 2 -> b

    acc = ct.zeros((m0, n0), dtype=ct.float32)

    # SEQ-Schleife laeuft ueber dim 5 (s), die PRIM-K-Dimension (p) steckt in dim 8
    for k_i in range(k):
        # M steht in dim 6, K in dim 8 -> Tile ist bereits (m, k), kein transpose noetig
        A_ = ct.load(
            A,
            index=(m3_i, m2_i, n2_i, m1_i, n1_i, k_i, 0, 0, 0),
            shape=(1,1,1,1,1,1,m0,1,k0),
            padding_mode=ct.PaddingMode.ZERO
        )
        A_ = ct.reshape(A_, (m0, k0))
        # N steht in dim 7, K in dim 8 -> Tile ist (n, k), muss auf (k, n) transponiert werden
        B_ = ct.load(
            B,
            index=(m3_i, m2_i, n2_i, m1_i, n1_i, k_i, 0, 0, 0),
            shape=(1,1,1,1,1,1,1,n0,k0),
            padding_mode=ct.PaddingMode.ZERO
        )
        B_ = ct.reshape(B_, (n0, k0))
        B_ = ct.transpose(B_)
        acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (1,1,1,1,1,1,m0,n0,1))
    ct.store(C, index=(m3_i, m2_i, n2_i, m1_i, n1_i, 0, 0, 0, 0), tile=acc)



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

    tensor_acspx_32 = tensor_acspx.to('cuda')
    tensor_bspy_32 = tensor_bspy.to('cuda')
    
    tensor_acspx_16 = tensor_acspx.to('cuda').to(torch.float16)
    tensor_bspy_16 = tensor_bspy.to('cuda').to(torch.float16)

    config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    file_dir = Path(__file__).parent

    #print(config);
    # with open(file_dir / "results" / "task2_config.out", "w") as f:
    #     f.write(str(config))

    opti = Optimizer(config)
    print(opti.config)
    print(opti.make_executable())
    opti.split_dim(4, outer_size=None, inner_size=128)
    print(opti.make_executable())
    opti.split_dim(6, outer_size=None, inner_size=128)
    print(opti.make_executable())
    #opti.permute_dims([0, 1, 3, 2, 4, 7, 5, 6])
    #print(opti.config)


    # 1. Prepare Tensor A
    # tensor_acspx_16 is already in the correct physical memory layout.
    # We just map it to the 9D layout required by the opti.config.
    A_opt = torch.as_strided(
        tensor_acspx_16, 
        size=opti.config.dim_sizes, 
        stride=opti.config.strides[0]
    )

    # 2. Prepare Tensor B
    # Strides[1] = [0, 0, 4718592, 0, 128, 73728, 0, 1, 1152]:
    # y stride 1, p stride 1152 (=y), s stride 73728 (=p*y), b stride 4718592 (=s*p*y)
    # -> die Config erwartet (b, s, p, y) contiguous, also tensor_bspy_16 unveraendert.
    B_opt = torch.as_strided(
        tensor_bspy_16,
        size=opti.config.dim_sizes,
        stride=opti.config.strides[1]
    )

    # 3. Prepare Tensor C
    # Strides[2] = [21233664, 1769472, 5308416, 128, 196608, 0, 1, 1536, 0]:
    # x stride 1, y stride 1536 (=x), c stride 1769472 (=y*x),
    # b stride 5308416 (=c*y*x), a stride 21233664 (=b*c*y*x)
    # -> die Config schreibt direkt im Ziel-Layout (a, b, c, y, x).
    # Sizes: a=4, b=4, c=3, y=1152, x=1536
    C_abcyx = torch.empty((4, 4, 3, 1152, 1536), dtype=torch.float16, device='cuda')
    C_opt = torch.as_strided(
        C_abcyx,
        size=opti.config.dim_sizes,
        stride=opti.config.strides[2]
    )

    # PAR-Dims sind 0..4 (a, c, b, x2, y2); 0 und 1 bekommen eigene Grid-Slots,
    # 2..4 werden im Kernel aus bid(2) zurueckgerechnet.
    grid = (opti.config.dim_sizes[0], opti.config.dim_sizes[1], opti.config.dim_sizes[2] * opti.config.dim_sizes[3] * opti.config.dim_sizes[4])

    kernel_args = (
        A_opt, B_opt, C_opt,
        opti.config.dim_sizes[3],   # m1 = x2 = 12
        opti.config.dim_sizes[4],   # n1 = y2 = 9
        opti.config.dim_sizes[5],   # k  = s  = 64  (SEQ-Schleife)
        opti.config.dim_sizes[6],   # m0 = x1 = 128
        opti.config.dim_sizes[7],   # n0 = y1 = 128
        opti.config.dim_sizes[8],   # k0 = p  = 64  (PRIM-K)
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        contraction,
        kernel_args
    )

    C_final = C_abcyx

    expected = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)
    assert torch.allclose(C_final, expected, atol=1e-2), "The result is incorrect!"
    print("The result is correct!")

    plot_tensor(
        C_final.to('cpu'),
        path=file_dir / 'results' / 'try3_16.png',
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
        kernel_args
    ))
    
    tflops_opt = flops / (t_ms_opt / 1000) / (10**12)
    
    print(f"\nOptimized kernel:")
    print(f"Execution time of optimized kernel: {t_ms_opt:.2f} ms")
    print(f"TFLOPS of optimized kernel: {tflops_opt:.2f}")

