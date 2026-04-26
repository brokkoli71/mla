import cuda.tile as ct
import cupy as cp
import torch
import triton
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@ct.kernel
def kernel_matmul(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_y, k_dim):
    pid = ct.bid(0)
    
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

    fp = lambda: ct.launch(torch.cuda.current_stream(), grid, kernel_matmul, (A, B, C, tm, tn, tk, grid_y, K))
    
    if check_correctness:
        # Run once to populate C
        fp()
        vgl = torch.matmul(A, B)
        assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1e-1), f"Incorrect result for tile shape ({tm}, {tn}, {tk})"

    # Benchmark
    # Adjust warmup and rep for extremely large sizes to save execution time
    warmup = 10 if M >= 4096 else 25
    rep = 100 if M >= 4096 else 500
    
    t_ms = triton.testing.do_bench(fp, warmup=warmup, rep=rep)
    
    # Calculate TFLOPs
    tflops = (2 * M * N * K) / (t_ms * 1e-3 * 1e12)
    return tflops

def task_3a():
    print("--- Running Task 3a: Scaling Matrix Sizes ---")
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    tm, tn, tk = 64, 64, 64
    
    tflops_results = []
    
    for size in sizes:
        try:
            tflops = run_benchmark(size, size, size, tm, tn, tk)
            print(f"Matrix: {size}x{size}x{size} | Tile: ({tm}, {tn}, {tk}) -> {tflops:.2f} TFLOPS")
            tflops_results.append(tflops)
        except Exception as e:
            print(f"Failed for size {size}: {e}")
            tflops_results.append(0.0)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, tflops_results, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title("Task 3a: Kernel Performance vs Matrix Size\nTile Size = (64, 64, 64)")
    plt.xlabel("Matrix Dimension (M = N = K)")
    plt.ylabel("Performance (TFLOPS)")
    plt.xscale('log', base=2)
    plt.xticks(sizes, labels=[str(s) for s in sizes])
    plt.grid(True, which="both", ls="--")
    file_dir = Path(__file__).parent
    plt.savefig(file_dir / "task_3a_scaling.png", bbox_inches='tight')
    plt.close()
    print("Saved plot to 'task_3a_scaling.png'\n")

def task_3b():
    print("--- Running Task 3b: Tile Shape Search ---")
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
        plt.title(f"Task 3b: Heatmap for Matrix {size}x{size}x{size}\n(Fixed k_tile = 64)", pad=15)
        
        # Annotate cells with values
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{heatmap_data[i, j]:.1f}", ha='center', va='center', color='white' if heatmap_data[i, j] < heatmap_data.max()*0.7 else 'black')
                
        file_dir = Path(__file__).parent
        plt.savefig(file_dir / f"task_3b_heatmap_{size}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved heatmap to 'task_3b_heatmap_{size}.png'")

if __name__ == "__main__":
    task_3a()
    task_3b()