import cuda.tile as ct
import cupy as cp
import torch
import triton

def main():
    A_16 = torch.randn((64, 4096), device='cuda', dtype=torch.float16)
    B_16 = torch.randn((4096, 64), device='cuda', dtype=torch.float16)
    C = torch.empty((64, 64), device='cuda', dtype=torch.float32)


    fp_16 = lambda : ct.launch(torch.cuda.current_stream(), (1,), kernel_fp16, (A_16, B_16, C))
    t = triton.testing.do_bench(fp_16, warmup=25, rep=1000)
    tflops = (2* 64 * 4096 * 64) / (t * 1e-3 * 1e12)
    print("16bit TFLOPs: ", tflops)

    #fp_16_bad = lambda : ct.launch(torch.cuda.current_stream(), (1,), kernel_fp16_bad, (A, B, C1))
    #t1 = triton.testing.do_bench(fp_16_bad, warmup=1, rep=1)
    #print(t1)

    A_32 = torch.randn((64, 4096), device='cuda', dtype=torch.float32)
    B_32 = torch.randn((4096, 64), device='cuda', dtype=torch.float32)
    C1 = torch.empty((64, 64), device='cuda', dtype=torch.float32)

    fp_32 = lambda : ct.launch(torch.cuda.current_stream(), (1,), kernel_fp32, (A_32, B_32, C1))
    t = triton.testing.do_bench(fp_32, warmup=25, rep=1000)
    tflops = (2* 64 * 4096 * 64) / (t * 1e-3* 1e12)
    print("32bit TFLOPs: ", tflops)

    vgl = torch.matmul(A_16, B_16)
    vgl_32 = torch.matmul(A_32, B_32)
    

    # print(C[:5,:5])
    # print(C1[:5,:5])
    # print(vgl[:5,:5])

    assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1e-1), "The 16 result is incorrect!"
    assert torch.allclose(C1, vgl_32, atol=1e-1), "The 32 result is incorrect!"


@ct.kernel
def kernel_fp16(A, B, C):

    pid = ct.bid(0)
    accumulator = ct.full((64, 64), 0, dtype=ct.float32)


    for k in range(64):
        
        a = ct.load(A, index=(pid, k), shape=(64, 64), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, pid), shape=(64, 64), padding_mode=ct.PaddingMode.ZERO)
    
        
        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(0,0), tile=accumulator)

@ct.kernel
def kernel_fp32(A, B, C):

    pid = ct.bid(0)
    accumulator = ct.full((64, 64), 0, dtype=ct.float32)


    for k in range(64):
        
        a = ct.load(A, index=(pid, k), shape=(64, 64), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, pid), shape=(64, 64), padding_mode=ct.PaddingMode.ZERO)
    
        
        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(0,0), tile=accumulator)


@ct.kernel
def kernel_fp16_bad(A, B, C):

    pid = ct.bid(0)
    accumulator = ct.full((64, 64), 0, dtype=ct.float32)

    a = ct.load(
        A,
        index=(pid, 0), 
        shape=(64, 4096), 
        padding_mode=ct.PaddingMode.ZERO
    )

    b = ct.load(
        B,
        index=(pid, 0), 
        shape=(4096, 64), 
        padding_mode=ct.PaddingMode.ZERO
    )

    accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(0,0), tile=accumulator)


if __name__ == "__main__":
    main()