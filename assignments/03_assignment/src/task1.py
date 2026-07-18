import cuda.tile as ct
import cupy as cp
from numpy.strings import index
import torch
import triton

def main():
    times = []
    for dt in [torch.float16, torch.float32]:
        run(dt)
        t = benchmark(dt)
        print(f"runtime {dt}: {t}")
        times.append(t)
    print(f"speedup: {times[1]/times[0]}")

def setup(dt):
    inner_size = 4096
    A = torch.randn((64, inner_size), device='cuda', dtype=dt)
    B = torch.randn((inner_size, 64), device='cuda', dtype=dt)
    C = torch.empty((64, 64), device='cuda', dtype=torch.float32)
    
    grid = (1, )

    torch.cuda.init()
    return A, B, C, grid
def run(dt):
    A, B, C, grid = setup(dt)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C))
    torch.cuda.synchronize()

    expected = torch.empty((64, 64), device='cuda', dtype=dt)
    torch.matmul(A, B, out=expected)
    expected = expected.to(torch.float32)  # Convert to float32 for comparison
    assert torch.allclose(C, expected, atol=1e-1), "The result is incorrect!"

def benchmark(dt):
    A, B, C, grid = setup(dt)
    def run_kernel():
        ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C))
        torch.cuda.synchronize()

    t = triton.testing.do_bench(run_kernel)
    return t

@ct.kernel
def kernel(A, B, C):
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