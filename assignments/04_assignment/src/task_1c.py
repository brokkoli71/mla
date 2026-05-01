import math

import cuda.tile as ct
import cupy as cp
import torch
import triton

einsum_str = "eabklxy,ecklyz->eabcxz"
# M = abx, N = cz, K = kly, C = e

def main(
    a = 15,
    b = 15,
    x = 3,
    c = 4,
    z = 11,
    k = 33,
    l = 5,
    y = 81,
    e = 3
):
    x_padded = int(2**math.ceil(math.log2(x))) 
    y_padded = int(2**math.ceil(math.log2(y))) 
    z_padded = int(2**math.ceil(math.log2(z)))

    print(f"Tensor shapes: A: {(e,a,b,k,l,x_padded,y_padded)}, B: {(e,c,k,l,y_padded,z_padded)}, C: {(e,a,b,c,x_padded,z_padded)}")
    # assert not to big (32 GiB)
    size_float16 = 2
    max_size = 32 * 1024 * 1024 * 1024
    required_size = (e*a*b*k*l*x_padded*y_padded + e*c*k*l*y_padded*z_padded + e*a*b*c*x_padded*z_padded)*size_float16
    assert required_size < max_size, "The tensors are too big for the GPU memory!"

    print(f"Required memory: {required_size / (1024**3):.2f} GiB")
    A = torch.randn((e,a,b,k,l,x_padded,y_padded), device='cuda', dtype=torch.float16)
    B = torch.randn((e,c,k,l,y_padded,z_padded), device='cuda', dtype=torch.float16)
    C = torch.empty((e,a,b,c,x_padded,z_padded), device='cuda', dtype=torch.float16)

    grid = (e, a, c)

    torch.cuda.init()
    ct.launch(torch.cuda.current_stream(), grid, contraction, (A, B, C, k, l, x_padded, y_padded, z_padded, b))
    torch.cuda.synchronize()

    expected = torch.einsum(einsum_str, A, B)
    assert torch.allclose(C, expected, atol=1e-0), "The result is incorrect!"
    print(f"Success!")
    

@ct.kernel
def contraction(A, B, C, k: ct.Constant[int], l: ct.Constant[int], x: ct.Constant[int], y: ct.Constant[int], z: ct.Constant[int], b: ct.Constant[int]):
    e_it = ct.bid(0)
    a_it = ct.bid(1)
    c_it = ct.bid(2)

    for b_it in range(b):
        acc = ct.zeros((x, z), dtype=ct.float32)
        for k_it in range(k):
            for l_it in range(l):
                A_ = ct.load(
                    A, 
                    index=(e_it,a_it,b_it,k_it,l_it,0,0), 
                    shape=(1,1,1,1,1,x,y), 
                    padding_mode=ct.PaddingMode.ZERO
                )
                A_ = ct.reshape(A_, (x, y))
                B_ = ct.load(
                    B, 
                    index=(e_it,c_it,k_it,l_it,0,0), 
                    shape=(1,1,1,1,y,z), 
                    padding_mode=ct.PaddingMode.ZERO
                )
                B_ = ct.reshape(B_, (y, z))
                acc+= ct.matmul(A_, B_)
        acc = ct.astype(acc, ct.float16)
        acc = ct.reshape(acc, (1,1,1,1,x,z))
        ct.store(C, index=(e_it,a_it,b_it,c_it,0,0), tile=acc)


if __name__ == "__main__":
    main()