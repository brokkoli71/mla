import triton
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from task4 import run, setup
def main(N=[2**i for i in range (4, 18)], name="", log_scaling=True):
    M = 2048
    dtype = torch.float32
    bandwidths = []
    for n in N:
        grid, A, B, tile_M, tile_N = setup(M=M, N=n, tile_M=64, tile_N=n, dtype=dtype)
        t = triton.testing.do_bench(lambda: run(grid, A, B, tile_M, tile_N))
        assert torch.allclose(B, A), "Task 4 failed: B does not match A!"
        print(f"matrix_copy benchmark for N={n}: {t:.2f} ms")
        element_size = torch.tensor([], dtype=dtype).element_size()
        bandwidth = 2 * M * n * element_size / (t * 1e6)  # GB/s
        bandwidths.append(bandwidth)

    # plot results
    plt.plot(N, bandwidths, marker='o')
    plt.xlabel('N (number of columns)')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('matrix_copy Benchmark: Bandwidth vs N')
    if log_scaling:
        plt.xscale('log', base=2)
        plt.xticks(N)

    plt.grid(True)

    file_dir = Path(__file__).parent
    plt.savefig(file_dir / f'task4_benchmark{name}.png')
    plt.close()

if __name__ == "__main__":
    main()
    main(N=range(1, 129, 4), name="_fullrange", log_scaling=False)