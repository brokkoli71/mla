import cuda.tile as ct
import cupy as cp
import torch
import triton

def main():
    A = torch.randn((64, 4096), device='cuda', dtype=torch.float16)
    B = torch.randn((4096, 64), device='cuda', dtype=torch.float16)
    C = torch.empty((64, 64), device='cuda', dtype=torch.float32)
    
    torch.cuda.init()
    ct.launch(torch.cuda.current_stream(), (1,), kernel_fp16, (A, B, C))
    torch.cuda.synchronize()

    assert torch.allclose(C, torch.matmul(A, B), atol=1e-2), "The result is incorrect!"


@ct.kernel
def kernel_fp16(A, B, C):
    ct.mma(A, B, C)

if __name__ == "__main__":
    main()