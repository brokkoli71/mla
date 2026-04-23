import math

import cuda.tile as ct
import cupy as cp
from numpy.strings import index
import torch
import triton

def main():
    M = 32
    N = 32
    K = 32
    # TODO: use this padded sizes
    M_padded = int(2**math.ceil(math.log2(M))) 
    N_padded = int(2**math.ceil(math.log2(N)))
    K_padded = int(2**math.ceil(math.log2(K)))
    m_tile=4
    n_tile=4
    k_tile=4


    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    grid = ((M_padded // m_tile) * (N_padded // n_tile), )

    torch.cuda.init()
    ct.launch(torch.cuda.current_stream(), grid, kernel_fp16, (A, B, C, m_tile, n_tile, k_tile))
    torch.cuda.synchronize()

    expected = torch.empty((M, N), device='cuda', dtype=torch.float16)
    torch.matmul(A, B, out=expected)
    expected = expected.to(torch.float32)  # Convert to float32 for comparison
    # print("Expected:\n", expected)
    # print("Actual:\n", C)
    assert torch.allclose(C, expected, atol=1e-1), "The result is incorrect!"


@ct.kernel
def kernel_fp16(A, B, C, m_tile: ct.Constant[int], n_tile: ct.Constant[int], k_tile: ct.Constant[int]):
    bid = ct.bid(0)
    bid_x = bid % (C.shape[1] // n_tile)
    bid_y = bid // (C.shape[1] // n_tile)

    result = ct.zeros((m_tile, n_tile), dtype=torch.float32)
    for k in range(0, A.shape[1] // k_tile):
        A_block = ct.load(A, index=(bid_y , k), shape=(m_tile, k_tile), padding_mode=ct.PaddingMode.ZERO)
        B_block = ct.load(B, index=(k, bid_x), shape=(k_tile, n_tile), padding_mode=ct.PaddingMode.ZERO)
        result = ct.mma(A_block, B_block, acc=result)
    # print("Result in kernel:\n", result)
    ct.store(C, index=(bid_y, bid_x), tile=result)

if __name__ == "__main__":
    main()