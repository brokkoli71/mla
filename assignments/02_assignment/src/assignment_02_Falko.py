import torch
import cuda.tile as ct
import cupy as cp

@ct.kernel
def matrix_reduce(a, b, tile_size: ct.Constant[int]):

    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid, 0), shape=(1, tile_size,))
    #b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = ct.sum(a_tile, 1)

    ct.store(b, index=(pid, ), tile=result)


def task01():
    print(cp.cuda.Device().attributes.items())
    # ('L2CacheSize', 25165824)
    # ('MaxSharedMemoryPerMultiprocessor', 102400)
    # ('ClockRate', 2418000) 

def task02():
    m = 32
    k = 32
    #torch.seed(0)
    matrix_a = torch.randn((m,k), device='cuda')
    vec_b = torch.zeros(m, device='cuda')

    tile_size = k
    grid = (m, 1, 1 )

    # Launch kernel
    ct.launch(torch.cuda.current_stream().cuda_stream,
            grid, 
            matrix_reduce,
            (matrix_a, vec_b,  tile_size))


    # Verification
    torch_result = torch.sum(matrix_a, dim=1)
    if torch.allclose(vec_b, torch_result, atol=1e-5):
        print("Success!")
    else:
        print("Mismatch detected.")





def main():
    #task01()
    task02()



if __name__ == "__main__":
    main()
