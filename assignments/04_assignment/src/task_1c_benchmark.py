from task_1c import main as implementation_c
from task_1b import main as implementation_b
import torch
import triton.testing
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    a = 64
    c = 64
    k = 1
    l = 16
    xe_values = [(1, 64), (256, 1)]
    y = 32
    z = 32
    b = 8
    results_b = np.zeros(len(xe_values))
    results_c = np.zeros(len(xe_values))


    for j, (x, e) in enumerate(xe_values):
        A = torch.randn((e,a,b,k,l,x,y), device='cuda', dtype=torch.float16)
        B = torch.randn((e,c,k,l,y,z), device='cuda', dtype=torch.float16)
        C = torch.empty((e,a,b,c,x,z), device='cuda', dtype=torch.float16)
        t_ms = triton.testing.do_bench(lambda: implementation_b(a=a, b=b, x=x, c=c, z=z, k=k, l=l, y=y, e=e, A=A, B=B, C=C, verbose=False))    
        results_b[j] = t_ms
        t_ms = triton.testing.do_bench(lambda: implementation_c(a=a, b=b, x=x, c=c, z=z, k=k, l=l, y=y, e=e, A=A, B=B, C=C, verbose=False))    
        results_c[j] = t_ms
    fig, ax = plt.subplots()
    # bar width
    bar_width = 0.35
    # Set position of bar on X axis
    r1 = np.arange(len(xe_values))
    r2 = [x + bar_width for x in r1]
    # Make the plot
    ax.bar(r1, results_b, color='blue', width=bar_width, edgecolor='grey', label='Implementation B')
    ax.bar(r2, results_c, color='orange', width=bar_width, edgecolor='grey', label='Implementation C')
    # Add labels and title
    ax.set_xlabel('x and e values', fontweight='bold')
    ax.set_ylabel('Time (ms)', fontweight='bold')
    ax.set_title('Performance Comparison of Implementations B and C')
    ax.set_xticks([r + bar_width/2 for r in range(len(xe_values))])
    ax.set_xticklabels([f'(x={x}, e={e})' for x, e in xe_values])
    ax.legend()

    file_dir = Path(__file__).parent
    plt.savefig(file_dir / 'benchmark_1b_vs_1c.png')

if __name__ == "__main__":
    main()