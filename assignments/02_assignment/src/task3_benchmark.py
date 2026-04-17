import triton
import torch
import cuda.tile as ct
from task3 import tensor_add_KL, tensor_add_MN

M, N, K, L = 16, 128, 16, 128
size = (M, N, K, L)

def main():
    t = triton.testing.do_bench(tensor_add_KL_benchmark)
    print(f"tensor_add_KL benchmark: {t:.2f} ms")
    t = triton.testing.do_bench(tensor_add_MN_benchmark)
    print(f"tensor_add_MN benchmark: {t:.2f} ms")
    
def tensor_add_KL_benchmark():
    A = torch.randn(size, device='cuda')
    B = torch.randn(size, device='cuda')
    C = torch.empty(size, device='cuda')
    grid_kl = (M, N)
    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid_kl, 
        tensor_add_KL,
        (A, B, C, K, L),)

def tensor_add_MN_benchmark():
    A = torch.randn(size, device='cuda')
    B = torch.randn(size, device='cuda')
    C = torch.empty(size, device='cuda')
    grid_mn = (K, L)
    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid_mn, 
        tensor_add_MN,
        (A, B, C, M, N),)


if __name__ == "__main__":
    main()