import torch
import cuda.tile as ct

def main():
    M, N, K, L = 16, 128, 16, 128
    size = (M, N, K, L)
    A = torch.randn(size, device='cuda')
    B = torch.randn(size, device='cuda')
    C_kl = torch.empty(size, device='cuda')
    C_mn = torch.empty(size, device='cuda')
    grid_kl = (M, N)
    grid_mn = (K, L)
    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid_kl, 
        tensor_add_KL,
        (A, B, C_kl, K, L),)
    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid_mn, 
        tensor_add_MN,
        (A, B, C_mn, M, N),)
    
    expected = A + B
    assert torch.allclose(C_kl, expected), "Task 3 KL failed: C_kl does not match expected result!"
    assert torch.allclose(C_mn, expected), "Task 3 MN failed: C_mn does not match expected result!"

def tensor_add(A, B, C, index, shape):
    A_block = ct.load(
            A, 
            index=index,
            shape=shape,
        )
    B_block = ct.load(
            B, 
            index=index,
            shape=shape,
        )

    result = A_block + B_block

    ct.store(C, index=index, tile=result)

@ct.kernel
def tensor_add_KL(A, B, C, K: ct.Constant[int], L: ct.Constant[int]):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    # Each block processes the full (K, L) sub-tensor at a specific (m, n)
    index = (bid_m, bid_n, 0, 0)
    shape = (1, 1, K, L)
    tensor_add(A, B, C, index, shape)

@ct.kernel
def tensor_add_MN(A, B, C, M: ct.Constant[int], N: ct.Constant[int]):
    bid_k = ct.bid(0)
    bid_l = ct.bid(1)
    
    # Each block processes the full (M, N) sub-tensor at a specific (k, l)
    index = (0, 0, bid_k, bid_l)
    shape = (M, N, 1, 1)
    
    tensor_add(A, B, C, index, shape)    

if __name__ == "__main__":
    main()