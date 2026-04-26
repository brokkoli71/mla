import cuda.tile as ct
import cupy as cp
import torch
import triton
import math


def main():


    A = torch.randn((513, 4096), device='cuda', dtype=torch.float16)
    B = torch.randn((4096, 513), device='cuda', dtype=torch.float16)

    C = torch.empty((513, 513), device='cuda', dtype=torch.float32)

    tm = 128
    tn = 128
    tk = 64

    m, k = A.shape
    _, n = B.shape

    grid_x = math.ceil(m / tm)
    grid_y = math.ceil(n / tn)
    grid = (grid_x * grid_y, 1, 1)

    fp = lambda : ct.launch(torch.cuda.current_stream(), grid, kernel_matmul, (A, B, C, tm, tn, tk, grid_y, k))
    t = triton.testing.do_bench(fp, warmup=25, rep=1000)
    tflops = (2* m * k * n) / (t * 1e-3* 1e12)
    print("TFLOPs: ", tflops)
    
    vgl = torch.matmul(A, B)
    # print(C[:5,:5])
    # print(vgl[:5,:5])

    assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1), "The result is incorrect!"


@ct.kernel
def kernel_matmul(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_y, k_dim):

    pid = ct.bid(0)

    #num_tiles_k = math.ceil(k_dim / tk)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)

    pid_m = pid // grid_y
    pid_n = pid % grid_y

    for k in range(num_tiles_k):
        
        a = ct.load(A, index=(pid_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, pid_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)

        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(pid_m, pid_n), tile=accumulator)


if __name__ == "__main__":
    main()