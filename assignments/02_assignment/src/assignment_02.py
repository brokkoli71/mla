import torch
import cuda.tile as ct
import cupy as cp
import numpy as np
import math

def main():
    cp.cuda.Device().attributes.items()
    print("CUDA Device Attributes:" )
    for key, value in cp.cuda.Device().attributes.items():
        if key in ["L2CacheSize", "MaxSharedMemoryPerMultiprocessor", "ClockRate"]:
            print(f"\t{key}: {value}")
    
    task_1()

def task_1():
    M = 65
    K = 33
    
    # Calculate the next power of two for K
    TILE_K = int(2**math.ceil(math.log2(K))) # This will equal 64
    
    A = torch.randn(M, K, device='cuda')
    A_reduced = torch.empty(M, device='cuda')

    grid = (M,)

    # Launch kernel, passing TILE_K as the constant
    ct.launch(torch.cuda.current_stream().cuda_stream,
    grid, 
    matrix_reduce,
    (A, A_reduced, TILE_K))

    expected = torch.sum(A, dim=1)
    assert torch.allclose(A_reduced, expected), "Task 1 failed: A_reduced does not match expected result!"
    print("Task 1 passed!")

@ct.kernel
def matrix_reduce(A, A_reduced, TILE_K: ct.Constant[int]):
    pid = ct.bid(0)

    # Use the power-of-two TILE_K for the shape. 
    # The padding mode handles the difference between K (33) and TILE_K (64).
    A_row = ct.load(
            A, 
            index=(pid, 0), 
            shape=(1, TILE_K), 
            padding_mode=ct.PaddingMode.ZERO
        )
    
    result = ct.sum(A_row, axis=1)

    ct.store(A_reduced, index=(pid, ), tile=result)

if __name__ == "__main__":
    main()