import math

from config import ExecType, generate_config
from pathlib import Path
import torch
import cupy as cp
from optimizer import Optimizer
import cuda.tile as ct


def power_of_two_contained_in(n):
    """Returns the largest power of two that is less than or equal to n."""
    if n < 1:
        raise ValueError("Input must be a positive integer.")
    power = 1
    while power * 2 <= n:
        power *= 2
    return power

def task_a_and_b():
    einsum = "cmk, ckn -> cmn"
    c = 4 
    m = n = k = 4096
    input_shapes = [(c, m, k), (c, k, n)]
    config = generate_config(einsum, input_shapes, "cmkn")
    file_dir = Path(__file__).parent
    with open(file_dir / "task_4a.out", "w") as f:
        f.write(str(config))

    L2_cache_size = cp.cuda.Device().attributes["L2CacheSize"]
    max_shared_memory_per_multiprocessor = cp.cuda.Device().attributes["MaxSharedMemoryPerMultiprocessor"]
    max_shared_memory_per_block = cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"]
    print(f"L2 cache size: {L2_cache_size / (1024**2)} MB")
    print(f"Max shared memory per multiprocessor: {max_shared_memory_per_multiprocessor / 1024} KB")
    print(f"Max shared memory per block: {max_shared_memory_per_block / 1024} KB")

    data_type_size_fp32 = torch.tensor([], dtype=torch.float32).element_size()
    data_type_size_fp16 = torch.tensor([], dtype=torch.float16).element_size()
    print(f"Data type size (FP32): {data_type_size_fp32} bytes")
    print(f"Data type size (FP16): {data_type_size_fp16} bytes")

    optimizer = Optimizer(config) #c,m,k,n
    # RAM Usage \approx (m_prim * k_prim + k_prim * n_prim) * data_type_size

    m_prim = n_prim = power_of_two_contained_in(math.sqrt(max_shared_memory_per_block / 12))
    k_prim = power_of_two_contained_in((max_shared_memory_per_block - (4 * m_prim * m_prim)) // (4 * m_prim))
    print(f"RAM usage for tile size {m_prim}x{k_prim} and {k_prim}x{n_prim}: {(m_prim * k_prim + k_prim * n_prim) * data_type_size_fp32 / 1024} KiB")

    # for l2 cache we want all the matrix [|m_l2|, |n_l2|, |m_prim|, |n_prim|, |k_prim|]
    prim_size = (m_prim * k_prim + k_prim * n_prim) * data_type_size_fp16 + (m_prim * n_prim) * data_type_size_fp32
    # k is sequentially
    kernel_size = prim_size * (k // k_prim)
    kernels_per_l2_cache = L2_cache_size // kernel_size
    print(f"Kernels per L2 cache: {kernels_per_l2_cache}")
    m_l2 = power_of_two_contained_in(math.sqrt(kernels_per_l2_cache))
    n_l2 = power_of_two_contained_in(kernels_per_l2_cache // m_l2)

    optimizer.split_dim(3, inner_size=n_prim) # -> c,m,k,n,nprim
    optimizer.split_dim(3, inner_size=n_l2) # -> c,m,k,n,n_l2,n_prim
    optimizer.split_dim(2, inner_size=k_prim) # -> c,m,k,k_prim,n,n_l2,n_prim
    optimizer.split_dim(1, inner_size=m_prim) # -> c,m,m_prim,k,k_prim,n,n_l2,n_prim
    optimizer.split_dim(1, inner_size=m_l2) # -> c,m,m_l2,m_prim,k,k_prim,n,n_l2,n_prim
    optimizer.permute_dims([0, 1, 6, 2, 7, 4, 3, 8, 5]) # -> c,m,n,m_l2,n_l2,k,m_prim,k_prim,n_prim
    optimizer.make_executable()
    print(optimizer.config)

    grid_dims = [size for exec_type, size in 
                    zip(optimizer.config.exec_types, optimizer.config.dim_sizes) 
                    if exec_type == ExecType.PAR]
    print(f"Grid dimensions: {grid_dims}")
    def product(lst):
        result = 1
        for num in lst:
            result *= num
        return result
    grid = (grid_dims[0], grid_dims[1], product(grid_dims[2:]))
    print(f"Grid: {grid}")


@ct.kernel
def muliply_kernel():
    pass

if __name__ == "__main__":
    task_a()