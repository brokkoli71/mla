import triton
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from task4 import main as main_task4
def main():
    N = [16, 32, 64, 128, 256, 512, 1024, 2048]
    M = 2048
    dtype = torch.float32
    bandwidths = []
    for n in N:
        t = triton.testing.do_bench(lambda: main_task4(M=M, N=n, tile_M=64, tile_N=n, dtype=dtype))
        print(f"matrix_copy benchmark for N={n}: {t:.2f} ms")
        element_size = torch.tensor([], dtype=dtype).element_size()
        bandwidth = 2 * M * n * element_size / (t * 1e6)  # GB/s
        bandwidths.append(bandwidth)

    # plot results
    plt.plot(N, bandwidths, marker='o')
    plt.xlabel('N (number of columns)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('matrix_copy Benchmark: Bandwidth vs N')
    plt.xscale('log', base=2)
    plt.xticks(N)
    plt.grid(True)

    file_dir = Path(__file__).parent
    plt.savefig(file_dir / 'task4_benchmark.png')

if __name__ == "__main__":
    main()