import cuda.tile as ct
import torch
import triton
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from task3 import run_benchmark as run_benchmark_task2

def main():
    plot_order()

    A = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    B = torch.randn((4096, 8192), device='cuda', dtype=torch.float16)

    C = torch.empty((8192, 8192), device='cuda', dtype=torch.float32)

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
    assert torch.allclose(C, vgl.to(dtype=torch.float32), atol=1), "The result is incorrect!"

    task_4b()

def calc_position(pid, swizzle_size, grid_y, grid_x):
    num_pid_in_stripe = swizzle_size * grid_x
    stripe_index = pid // num_pid_in_stripe
    begin_n = stripe_index * swizzle_size

    stripe_height = swizzle_size
    if (begin_n + swizzle_size) > grid_y:
        stripe_height = grid_y - begin_n

    index_n_temp = pid % stripe_height
    index_m_temp = pid // stripe_height

    index_m = index_m_temp % grid_x
    index_n = begin_n + index_n_temp

    return index_m, index_n

def plot_order():
    # draw the execution order of PIDs for a grid whose grid_y is *not* a
    # multiple of swizzle_size, so the trailing (clamped) stripe is exercised too
    swizzle_size = 4
    grid_x = 8
    grid_y = 9
    num_pids = grid_x * grid_y

    fig, ax = plt.subplots(figsize=(8, 8))
    last_pos = None
    for pid in range(num_pids):
        index_m, index_n = calc_position(pid, swizzle_size, grid_y, grid_x)

        if last_pos is not None:
            ax.arrow(last_pos[0] + 1 / 2, last_pos[1] + 1 / 2,
                      index_m - last_pos[0], index_n - last_pos[1],
                      head_width=0.15, head_length=0.15, length_includes_head=True,
                      fc='blue', ec='blue', linewidth=0.6)
        last_pos = (index_m, index_n)

    ax.set_xlim(-0.5, grid_x + 0.5)
    ax.set_ylim(-0.5, grid_y + 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Swizzled Execution Order (swizzle_size={swizzle_size}, grid={grid_x}x{grid_y})")
    ax.set_xlabel("Index M")
    ax.set_ylabel("Index N")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.savefig(__file__.replace('.py', '_execution_order.png'), bbox_inches='tight')
    plt.close(fig)

@ct.kernel
def kernel_matmul_swizzle(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int], grid_x, grid_y):

    swizzle_size = 8
    pid = ct.bid(0)

    index_m, index_n = calc_position(pid, swizzle_size, grid_y, grid_x)

    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)


    for k in range(num_tiles_k):                                                
        
        a = ct.load(A, index=(index_m, k), shape=(tm, tk), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(B, index=(k, index_n), shape=(tk, tn), padding_mode=ct.PaddingMode.ZERO)

        accumulator = ct.mma(a, b, accumulator)

    ct.store(C, index=(index_m, index_n), tile=accumulator)


def run_benchmark(M, N, K, tm, tn, tk):
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float32)

    grid_x = math.ceil(M / tm)
    grid_y = math.ceil(N / tn)
    grid = (grid_x * grid_y, 1, 1)

    fp = lambda: ct.launch(torch.cuda.current_stream(), grid, kernel_matmul_swizzle, (A, B, C, tm, tn, tk, grid_x, grid_y))
    
    t_ms = triton.testing.do_bench(fp)
    
    # Calculate TFLOPs
    tflops = (2 * M * N * K) / (t_ms * 1e-3 * 1e12)
    return tflops

def task_4b():
    print("--- Running Task 4b: Tile Shape Search ---")
    matrix_sizes = [512, 2048]
    tile_dims = [32, 64, 128]
    best_tile_shapes = ""

    # Setup for heatmaps
    tile_indices = {32: 0, 64: 1, 128: 2}
    
    for size in matrix_sizes:
        print(f"\nBenchmarking Matrix Size: {size}x{size}x{size}")
        best_tflops = 0.0
        best_shape = None
        
        # Array to store heatmap data (m_tile, n_tile) for k_tile = 64
        heatmap_data = np.zeros((3, 3)) 
        
        for tm, tn, tk in itertools.product(tile_dims, tile_dims, tile_dims):
            try:
                tflops = run_benchmark(size, size, size, tm, tn, tk)
                
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
        out = f"-> BEST tile shape for {size}x{size}x{size} is {best_shape} achieving {best_tflops:.2f} TFLOPS"
        print(out)
        best_tile_shapes += out + "\n"
        
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
        
    with open(__file__.replace('.py', '_best_tile_shapes.txt'), 'w') as f:
        f.write(best_tile_shapes)

    task_4b_compare_to_task2()


def task_4b_compare_to_task2():
    print("--- Running Task 4b: Swizzle (Task 4) vs Row-Major (Task 2) Comparison ---")
    M, N, K = 8192, 8192, 4096
    tm, tn, tk = 128, 128, 64

    task2_tflops = run_benchmark_task2(M, N, K, tm, tn, tk)
    swizzle_tflops = run_benchmark(M, N, K, tm, tn, tk)
    speedup = swizzle_tflops / task2_tflops

    out = (
        f"\nTask 4b comparison at M={M}, N={N}, K={K}, tile=({tm}, {tn}, {tk}):\n"
        f"-> Task 2 kernel (row-major BIDs): {task2_tflops:.2f} TFLOPS\n"
        f"-> Task 4 kernel (swizzled BIDs):  {swizzle_tflops:.2f} TFLOPS\n"
        f"-> Speedup from swizzling: {speedup:.2f}x\n"
    )
    print(out)

    with open(__file__.replace('.py', '_best_tile_shapes.txt'), 'a') as f:
        f.write(out)


if __name__ == "__main__":
    main()