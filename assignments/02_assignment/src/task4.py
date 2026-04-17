import torch
import cuda.tile as ct

# Note: *args will prevent positional arguments
def main(*args, M = 2048, N = 16, tile_M = 64, tile_N = None, dtype=torch.float32):
    if tile_N is None:
        tile_N = N

    A = torch.randn((M, N), device='cuda', dtype=dtype)
    B = torch.empty_like(A, device='cuda', dtype=dtype)
    
    assert M % tile_M == 0, "M must be divisible by tile_M"
    assert N % tile_N == 0, "N must be divisible by tile_N"
    grid = (M // tile_M, N // tile_N)

    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid, 
        matrix_copy,
        (A, B, tile_M, tile_N))
    
    assert torch.allclose(B, A), "Task 4 failed: B does not match A!"

@ct.kernel
def matrix_copy(A, B, tile_M: ct.Constant[int], tile_N: ct.Constant[int]):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    index = (bid_m, bid_n)

    A_block = ct.load(
            A, 
            index=index,
            shape=(tile_M, tile_N),
        )
    
    ct.store(B, index=index, tile=A_block)

if __name__ == "__main__":
    main()