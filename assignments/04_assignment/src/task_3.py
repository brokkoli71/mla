import cuda.tile as ct
import cupy as cp
import torch
import triton

einsum_str = "ackm,bcnk->abnm"
# M = am, N = bn, K = ck
def main(
    m = 2**4,
    n = 2**4,
    k = 2**4,
):
    a = 16
    b = 16
    c = 32
    print(f"Tensor shapes: A: {(a,c,k,m)}, B: {(b,c,n,k)}, C: {(a,b,n,m)}")
    # assert not to big (32 GiB)
    size_float16 = 2
    max_size = 32 * 1024 * 1024 * 1024
    required_size = (a*b*k*m + c*k*n + a*b*n*m)*size_float16
    assert required_size < max_size, "The tensors are too big for the GPU memory!"

    print(f"Required memory: {required_size / (1024**3):.2f} GiB")
    A = torch.randn((a,c,k,m), device='cuda', dtype=torch.float16)
    B = torch.randn((b,c,n,k), device='cuda', dtype=torch.float16)
    C = torch.empty((a,b,n,m), device='cuda', dtype=torch.float16)
    
    grid = (a, b, n) 

    torch.cuda.init()
    args = (A, B, C, k, m, n, c)
    t_ms = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, contraction, args))
    print(f"Execution time of fused kernel: {t_ms:.2f} ms")
    expected = torch.einsum(einsum_str, A, B)
    assert torch.allclose(C, expected, atol=1e-0), "The result is incorrect!"
    print(f"Success!")    

@ct.kernel
def contraction(A, B, C, k, m, n, c):
    pass


if __name__ == "__main__":
    main()