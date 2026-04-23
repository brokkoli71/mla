import cuda.tile as ct
import cupy as cp
from numpy.strings import index
import torch
import triton

def main():
    inner_size = 4096
    A = torch.randn((64, inner_size), device='cuda', dtype=torch.float16)
    B = torch.randn((inner_size, 64), device='cuda', dtype=torch.float16)
    C = torch.empty((64, 64), device='cuda', dtype=torch.float32)
    
    grid = (1, )

    torch.cuda.init()
    ct.launch(torch.cuda.current_stream(), grid, kernel_fp16, (A, B, C))
    torch.cuda.synchronize()

    expected = torch.empty((64, 64), device='cuda', dtype=torch.float16)
    torch.matmul(A, B, out=expected)
    expected = expected.to(torch.float32)  # Convert to float32 for comparison
    assert torch.allclose(C, expected, atol=1e-1), "The result is incorrect!"


@ct.kernel
def kernel_fp16(A, B, C):
    m_tile=64
    n_tile=64
    k_tile=64

    result = ct.load(C, index=(0, 0), shape=(m_tile, n_tile))
    for i in range(0, A.shape[0] // m_tile):
        for j in range(0, B.shape[1] // n_tile):
            for k in range(0, A.shape[1] // k_tile):
                A_block = ct.load(A, index=(i, k), shape=(m_tile, k_tile))
                B_block = ct.load(B, index=(k, j), shape=(k_tile, n_tile))
                result = ct.mma(A_block, B_block, acc=result)

    ct.store(C, index=(0, 0), tile=result)

if __name__ == "__main__":
    main()