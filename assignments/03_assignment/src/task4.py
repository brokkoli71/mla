import cuda.tile as ct
import cupy as cp
import torch
import triton
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():

    A = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    B = torch.randn((4096, 8192), device='cuda', dtype=torch.float16)

    C = torch.empty((8192, 8192), device='cuda', dtype=torch.float32)
    C1 = torch.empty((8192, 8192), device='cuda', dtype=torch.float32)

    tm = 128
    tn = 128
    tk = 64

    m, k = A.shape
    _, n = B.shape

    grid_x = math.ceil(m / tm)
    grid_y = math.ceil(n / tn)
    grid = (grid_x * grid_y, 1, 1)

    fp_swizz = lambda : ct.launch(torch.cuda.current_stream(), grid, kernel_matmul_swizzle, (A, B, C, tm, tn, tk, grid_x,  grid_y))
    t = triton.testing.do_bench(fp_swizz, warmup=25, rep=1000)
    tflops = (2* m * k * n) / (t * 1e-3* 1e12)
    print("swizzle_kernel TFLOPs: ", tflops)

    vgl = torch.matmul(A, B)
    # print(C[:5,:5])
    # print(vgl[:5,:5])
    assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1), "The result is incorrect!"

    fp = lambda : ct.launch(torch.cuda.current_stream(), grid, kernel_matmul, (A, B, C1, tm, tn, tk, grid_x,  grid_y))
    t = triton.testing.do_bench(fp, warmup=25, rep=1000)
    tflops = (2* m * k * n) / (t * 1e-3* 1e12)
    print("non_swizzle_kernel TFLOPs: ", tflops)
    
    task_4b()


@ct.kernel
def kernel_matmul_swizzle_only_8th(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_x, grid_y):

    #swizzle_Group_size = 8
    pid = ct.bid(0)

    num_pid_in_block =  8 * 8
    blocks_m = grid_x // 8
    blocks_n = grid_y // 8

    index_m_temp  = pid % 8
    index_n_temp = (pid // 8) % 8

    block_index = pid // num_pid_in_block

    m_block_row = (block_index % blocks_m) * 8
    index_m = m_block_row + index_m_temp
    
    n_block_col = (pid // (num_pid_in_block * blocks_m)) * 8
    index_n = n_block_col + index_n_temp
    


    # first_index_m = (index_n_temp // 8) * 8
    # index_m = (first_index_m + index_m_temp) % (grid_x * 8)

    # index_n_block = index_n_temp % 8
    # firs_index_n_block = (pid // threads_in_x_block) * 8
    # index_n = firs_index_n_block + index_n_temp
 

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)


    for k in range(num_tiles_k):                                                
        
        a = ct.load(A, index=(index_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, index_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)

        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(index_m, index_n), tile=accumulator)


@ct.kernel
def kernel_matmul_swizzle(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_x, grid_y):

    swizzle_size = 8
    pid = ct.bid(0)

    num_pid_in_block = swizzle_size * grid_y
    block_index = (pid // num_pid_in_block)
    
    begin_m = (block_index * swizzle_size)

    swizzle = swizzle_size

    if (begin_m + swizzle_size) > grid_x:
        swizzle = grid_x - begin_m

    index_m_temp  = pid % swizzle
    index_n_temp = pid // swizzle
    
    index_n = index_n_temp % grid_y
    index_m = begin_m + index_m_temp


    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)


    for k in range(num_tiles_k):                                                
        
        a = ct.load(A, index=(index_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, index_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)

        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(index_m, index_n), tile=accumulator)

@ct.kernel
def kernel_matmul(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_x, grid_y):

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

def run_benchmark(M, N, K, tm, tn, tk, check_correctness=False):
    """
    Helper function to benchmark the kernel for a specific matrix and tile size.
    """
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    grid_x = math.ceil(M / tm)
    grid_y = math.ceil(N / tn)
    grid = (grid_x * grid_y, 1, 1)

    fp = lambda: ct.launch(torch.cuda.current_stream(), grid, kernel_matmul_swizzle, (A, B, C, tm, tn, tk, grid_x, grid_y))
    
    if check_correctness:
        # Run once to populate C
        fp()
        vgl = torch.matmul(A, B)
        assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1), f"Incorrect result for tile shape ({tm}, {tn}, {tk})"

    # Benchmark
    # Adjust warmup and rep for extremely large sizes to save execution time
    warmup = 10 if M >= 4096 else 25
    rep = 100 if M >= 4096 else 500
    
    t_ms = triton.testing.do_bench(fp, warmup=warmup, rep=rep)
    
    # Calculate TFLOPs
    tflops = (2 * M * N * K) / (t_ms * 1e-3 * 1e12)
    return tflops

def task_4b():
    print("--- Running Task 4b: Tile Shape Search ---")
    matrix_sizes = [512, 2048]
    tile_dims = [32, 64, 128]
    
    # Setup for heatmaps
    tile_indices = {32: 0, 64: 1, 128: 2}
    
    for size in matrix_sizes:
        print(f"\nBenchmarking Matrix Size: {size}x{size}x{size}")
        best_tflops = 0.0
        best_shape = None
        
        # Array to store heatmap data (m_tile, n_tile) for k_tile = 64
        heatmap_data = np.zeros((3, 3)) 
        
        # Iterate over all 27 combinations
        for tm, tn, tk in itertools.product(tile_dims, tile_dims, tile_dims):
            try:
                # Do a quick correctness check on small size to avoid silent failures
                check_corr = (size == 512) 
                tflops = run_benchmark(size, size, size, tm, tn, tk, check_correctness=check_corr)
                
                # Save best overall shape
                if tflops > best_tflops:
                    best_tflops = tflops
                    best_shape = (tm, tn, tk)
                
                # Save specifically for the heatmap (k_tile = 64)
                if tk == 64:
                    heatmap_data[tile_indices[tm], tile_indices[tn]] = tflops
                    
            except Exception as e:
                print(f"Tile {tm}x{tn}x{tk} failed: {e}")
                if tk == 64:
                    heatmap_data[tile_indices[tm], tile_indices[tn]] = 0.0
        
        print(f"-> BEST tile shape for {size}x{size}x{size} is {best_shape} achieving {best_tflops:.2f} TFLOPS")
        
        # Plot Heatmap
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(heatmap_data, cmap='viridis')
        fig.colorbar(cax, label='TFLOPS')
        
        # Labels and formatting
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(tile_dims)
        ax.set_yticklabels(tile_dims)
        ax.set_xlabel("n_tile")
        ax.set_ylabel("m_tile")
        ax.xaxis.set_ticks_position('bottom')
        plt.title(f"Task 4b: Heatmap for Matrix {size}x{size}x{size}\n(Fixed k_tile = 64)", pad=15)
        
        # Annotate cells with values
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color='white' if heatmap_data[i, j] < heatmap_data.max()*0.7 else 'black')
                
        file_dir = Path(__file__).parent
        plt.savefig(file_dir / f"task_4b_heatmap_{size}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to 'task_4b_heatmap_{size}.png'")


if __name__ == "__main__":
    main()