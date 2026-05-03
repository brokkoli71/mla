import math

import cuda.tile as ct
import cupy as cp
import torch
import triton

einsum_str = "eabklxy,ecklyz->eabcxz"
# M = abx, N = cz, K = kly, C = e

def main(
    a = 15,
    b = 104,
    x = 3,
    c = 41,
    z = 11,
    k = 33,
    l = 5,
    y = 11,
    e = 16,
    A = None,
    B = None,
    C = None,
    verbose = True
):
    x_padded = int(2**math.ceil(math.log2(x))) 
    y_padded = int(2**math.ceil(math.log2(y))) 
    z_padded = int(2**math.ceil(math.log2(z)))
    e_padded = int(2**math.ceil(math.log2(e)))

    # assert not to big (32 GiB)
    size_float16 = 2
    max_size = 32 * 1024 * 1024 * 1024
    required_size = (e_padded*a*b*k*l*x_padded*y_padded + e_padded*c*k*l*y_padded*z_padded + e_padded*a*b*c*x_padded*z_padded)*size_float16
    assert required_size < max_size, "The tensors are too big for the GPU memory!"

    if verbose:
        print(f"Tensor shapes: A: {(e_padded,a,b,k,l,x_padded,y_padded)}, B: {(e_padded,c,k,l,y_padded,z_padded)}, C: {(e_padded,a,b,c,x_padded,z_padded)}")
        print(f"Required memory: {required_size / (1024**3):.2f} GiB")
    if A is None:
        A = torch.randn((e_padded,a,b,k,l,x_padded,y_padded), device='cuda', dtype=torch.float16)
    if B is None:
        B = torch.randn((e_padded,c,k,l,y_padded,z_padded), device='cuda', dtype=torch.float16)
    if C is None:
        C = torch.empty((e_padded,a,b,c,x_padded,z_padded), device='cuda', dtype=torch.float16)

    grid = (a, b, c)
    
    torch.cuda.init()
    ct.launch(torch.cuda.current_stream(), grid, contraction, (A, B, C, k, l, x_padded, y_padded, z_padded, c, e_padded))
    torch.cuda.synchronize()

    expected = torch.einsum(einsum_str, A, B)
    assert torch.allclose(C, expected, atol=1e-0), "The result is incorrect!"
    if verbose:
        t_ms = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, contraction, (A, B, C, k, l, x_padded, y_padded, z_padded, c, e_padded)))
        print(f"Success! Time: {t_ms:.2f} ms")
    

@ct.kernel
def contraction(A, B, C, k: ct.Constant[int], l: ct.Constant[int], x: ct.Constant[int], y: ct.Constant[int], z: ct.Constant[int], c: ct.Constant[int], e: ct.Constant[int]):
    a_it = ct.bid(0)
    b_it = ct.bid(1)
    c_it = ct.bid(2)

    acc = ct.zeros((e, x, z), dtype=ct.float32)

    for k_it in range(k):
        for l_it in range(l):
            A_ = ct.load(
                A, 
                index=(0,a_it,b_it,k_it,l_it,0,0), 
                shape=(e,1,1,1,1,x,y), 
                padding_mode=ct.PaddingMode.ZERO
            )
            A_ = ct.reshape(A_, (e, x, y))
            B_ = ct.load(
                B, 
                index=(0,c_it,k_it,l_it,0,0), 
                shape=(e,1,1,1,y,z), 
                padding_mode=ct.PaddingMode.ZERO
            )
            B_ = ct.reshape(B_, (e, y, z))
            acc += ct.matmul(A_, B_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (e,1,1,1,x,z))
    ct.store(C, index=(0,a_it,b_it,c_it,0,0), tile=acc)

if __name__ == "__main__":
    main()