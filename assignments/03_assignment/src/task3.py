from task2 import run_kernel
import torch
import triton
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    m_tile=64
    n_tile=64
    k_tile=64

    TFLOPS_A = []

    for M in [256, 512, 1024, 2048, 4096, 8192]:
        A = torch.randn((M, M), device='cuda', dtype=torch.float16)
        B = torch.randn((M, M), device='cuda', dtype=torch.float16)
        C = torch.empty((M, M), device='cuda', dtype=torch.float32)
        t = triton.testing.do_bench(lambda: run_kernel(A, B, C, m_tile, n_tile, k_tile, M, M, M))
        TFLOPS = 2 * M * M * M / t / 1e9
        TFLOPS_A.append((M, TFLOPS))
        print(f"M={M}, TFLOPS={TFLOPS:.2f} with t={t:.2f} ms")

    # plot results
    M_values, TFLOPS_values = zip(*TFLOPS_A)
    plt.plot(M_values, TFLOPS_values, marker='o')
    plt.xlabel('M (matrix size)')
    plt.ylabel('TFLOPS')
    plt.title('Matrix Multiplication Performance: TFLOPS vs M')
    plt.xscale('log', base=2)
    plt.xticks(M_values)
    plt.grid(True)
    file_dir = Path(__file__).parent
    plt.savefig(file_dir / 'task3_a.png')


    TFLOPS_B = []

    for size in [2048, 512]:
        for m_tile in [32, 64, 128]:
            for n_tile in [32, 64, 128]:
                for k_tile in [32, 64, 128]:
                    A = torch.randn((size, size), device='cuda', dtype=torch.float16)
                    B = torch.randn((size, size), device='cuda', dtype=torch.float16)
                    C = torch.empty((size, size), device='cuda', dtype=torch.float32)
                    t = triton.testing.do_bench(lambda: run_kernel(A, B, C, m_tile, n_tile, k_tile, size, size, size))
                    TFLOPS = 2 * size * size * size / t / 1e9
                    TFLOPS_B.append((size, m_tile, n_tile, k_tile, TFLOPS))
                    print(f"Size={size}, m_tile={m_tile}, n_tile={n_tile}, k_tile={k_tile}, TFLOPS={TFLOPS:.2f} with t={t:.2f} ms")

    # save results to a file
    with open(file_dir / 'task3_b_results.txt', 'w') as f:
        for size, m_tile, n_tile, k_tile, TFLOPS in TFLOPS_B:
            f.write(f"Size={size}, m_tile={m_tile}, n_tile={n_tile}, k_tile={k_tile}, TFLOPS={TFLOPS:.2f}\n")
    
    # report best performing config per size to file
    best_configs = {}
    for size, m_tile, n_tile, k_tile, TFLOPS in TFLOPS_B:
        if size not in best_configs or TFLOPS > best_configs[size][4]:
            best_configs[size] = (size, m_tile, n_tile, k_tile, TFLOPS)
    with open(file_dir / 'task3_b_best_configs.txt', 'w') as f:
        for size, m_tile, n_tile, k_tile, TFLOPS in best_configs.values():
            report = f"Best config for Size={size}: m_tile={m_tile}, n_tile={n_tile}, k_tile={k_tile}, TFLOPS={TFLOPS:.2f}"
            print(report)
            f.write(report + '\n')

    # plot results as heatmap with m_tile on one axis and n_tile on the other, fixing k_tile = 64.
    for size in [2048, 512]:
        heatmap_data = [[0 for _ in range(3)] for _ in range(3)]
        for s, m_tile, n_tile, k_tile, TFLOPS in TFLOPS_B:
            if s == size and k_tile == 64:
                heatmap_data[[32, 64, 128].index(m_tile)][[32, 64, 128].index(n_tile)] = TFLOPS
        plt.figure()
        plt.imshow(heatmap_data, cmap='viridis', vmin=0)
        plt.colorbar(label='TFLOPS')
        plt.xticks([0, 1, 2], ['n_tile=32', 'n_tile=64', 'n_tile=128'])
        plt.yticks([0, 1, 2], ['m_tile=32', 'm_tile=64', 'm_tile=128'])
        plt.title(f'TFLOPS Heatmap for Size={size} with k_tile=64')
        plt.savefig(file_dir / f'task3_b_heatmap_size_{size}.png')


if __name__ == "__main__":
    main()