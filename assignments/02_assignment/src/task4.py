import torch
import cuda.tile as ct
import math
import triton

# Note: *args will prevent positional arguments
def setup(*args, M = 2048, N = 16, tile_M = 64, tile_N = None, dtype=torch.float32):
    if tile_N is None:
        tile_N = N

    tile_N = int(2**math.ceil(math.log2(tile_N))) 
    tile_M = int(2**math.ceil(math.log2(tile_M))) 

    A = torch.randn((M, N), device='cuda', dtype=dtype)
    B = torch.empty_like(A, device='cuda', dtype=dtype)
    
    grid = (math.ceil(M / tile_M), math.ceil(N / tile_N))

    return grid, A, B, tile_M, tile_N

def run(grid, A, B, tile_M, tile_N):
    ct.launch(torch.cuda.current_stream().cuda_stream,
        grid, 
        matrix_copy,
        (A, B, tile_M, tile_N))


@ct.kernel
def matrix_copy(A, B, tile_M: ct.Constant[int], tile_N: ct.Constant[int]):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    index = (bid_m, bid_n)

    A_block = ct.load(
            A, 
            index=index,
            shape=(tile_M, tile_N),
            padding_mode=ct.PaddingMode.ZERO
        )
    
    ct.store(B, index=index, tile=A_block)

def main():
    grid, A, B, tile_M, tile_N = setup()
    run(grid, A, B, tile_M, tile_N)
    assert torch.allclose(B, A), "Task 4 failed: B does not match A!"

if __name__ == "__main__":
    main()
