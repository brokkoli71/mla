import cuda.tile as ct
import cupy as cp
import torch
import triton
import math
import matplotlib.pyplot as plt
from pathlib import Path

einsum_str = "ackm,bcnk->abnm"
# M = am, N = bn, K = ck,  C=
def main(
    m = 16,
    n = 16,
    k = 16,
):
    a = 16
    b = 16
    c = 32

    m_padded = int(2**math.ceil(math.log2(m))) 
    n_padded = int(2**math.ceil(math.log2(n))) 
    k_padded = int(2**math.ceil(math.log2(k)))

    print(f"Tensor shapes: A: {(a,c,k,m_padded)}, B: {(b,c,n,k_padded)}, C: {(a,b,n,m_padded)}")
    # assert not to big (32 GiB)
    size_float16 = 2
    max_size = 32 * 1024 * 1024 * 1024
    required_size = (a*b*k*m_padded + c*k*n + a*b*n*m_padded)*size_float16
    assert required_size < max_size, "The tensors are too big for the GPU memory!"

    print(f"Required memory: {required_size / (1024**3):.2f} GiB")
    A = torch.randn((a,c,k,m_padded), device='cuda', dtype=torch.float16)
    B = torch.randn((b,c,n,k_padded), device='cuda', dtype=torch.float16)
    C = torch.empty((a,b,n,m_padded), device='cuda', dtype=torch.float16)

    grid = (a, b)

    torch.cuda.init()
    args = (A, B, C, m, n, k, c)
    t_ms = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, contraction, args))
    
    print(f"Execution time: {t_ms:.2f} ms")
    expected = torch.einsum(einsum_str, A, B)
    assert torch.allclose(C, expected, atol=1e-0), "The result is incorrect!"
    print(f"Success!")


    print("Starting Benchmarks...")
    
    sweep_range = list(range(17, 130)) # 17 bis 129
    
    # Sweep 1: Variiere n (m=64, k=64)
    times_n = []
    for n in sweep_range:
        t = run_contraction(m=64, n=n, k=64)
        times_n.append(t)
        
    # Sweep 2: Variiere k (m=64, n=64)
    times_k = []
    for k in sweep_range:
        t = run_contraction(m=64, n=64, k=k)
        times_k.append(t)

    print("Benchmarks completed. Creating plots...")

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1
    ax1.plot(sweep_range, times_n, marker='.', color='b')
    ax1.set_title("Sweep for Dimension n ($m=64, k=64$)")
    ax1.set_xlabel("size of dimension n")
    ax1.set_ylabel("execution time (ms)")
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Vertikale Linien für Zweierpotenzen
    for p in [32, 64, 128]:
        ax1.axvline(x=p, color='r', linestyle=':', alpha=0.5, label='Power of 2' if p==32 else "")
    ax1.legend()

    # Plot 2
    ax2.plot(sweep_range, times_k, marker='.', color='g')
    ax2.set_title("Sweep for Dimension k ($m=64, n=64$)")
    ax2.set_xlabel("size of dimension k")
    ax2.set_ylabel("execution time (ms)")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    for p in [32, 64, 128]:
        ax2.axvline(x=p, color='r', linestyle=':', alpha=0.5, label='Power of 2' if p==32 else "")
    ax2.legend()

    plt.tight_layout()
    file_dir = Path(__file__).parent
    plt.savefig(file_dir / 'benchmark_3.png')


# "ackm,bcnk->abnm"
@ct.kernel
def contraction(A, B, C, k: ct.Constant[int], m: ct.Constant[int], n: ct.Constant[int], c: ct.Constant[int]):
    a_it = ct.bid(0)
    b_it = ct.bid(1)

    acc = ct.zeros((n, m), dtype=ct.float32)

    for c_it in range(c):
        for k_it in range(k):
            A_ = ct.load(
                A, 
                index=(a_it, c_it, k_it, 0), 
                shape=(1, 1, 1, m), 
                padding_mode=ct.PaddingMode.ZERO
            )
            A_ = ct.reshape(A_, (1, m))
            B_ = ct.load(
                B, 
                index=(b_it, c_it , 0, k_it), 
                shape=(1, 1, n, 1), 
                padding_mode=ct.PaddingMode.ZERO
            )
            B_ = ct.reshape(B_, (n, 1))
            acc += ct.matmul(B_, A_)

    acc = ct.astype(acc, ct.float16)
    acc = ct.reshape(acc, (1, 1, n, m))
    ct.store(C, index=(a_it, b_it , 0, 0), tile=acc)

def run_contraction(m, n, k, a=16, b=16, c=32):
    # Nächste Zweierpotenz berechnen
    m_pad = int(2**math.ceil(math.log2(m))) 
    n_pad = int(2**math.ceil(math.log2(n))) 
    #k_pad = int(2**math.ceil(math.log2(k)))
    k_pad = k

    # Tensoren mit gepaddeten Dimensionen erstellen
    A = torch.randn((a, c, k_pad, m_pad), device='cuda', dtype=torch.float16)
    B = torch.randn((b, c, n_pad, k_pad), device='cuda', dtype=torch.float16)
    C_padded = torch.empty((a, b, n_pad, m_pad), device='cuda', dtype=torch.float16)

    grid = (a, b)
    args = (A, B, C_padded, k_pad, m_pad, n_pad, c)
    
    # Triton Benchmark durchführen (gibt Median-Zeit in ms zurück)
    ms = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, contraction, args))
    
    # Optionaler Correctness-Check (nur zur Sicherheit, kostet beim Benchmark keine Zeit, da do_bench vorher läuft)
    # C_final = C_padded[:, :, :n, :m] 
    
    return ms


if __name__ == "__main__":
    main()