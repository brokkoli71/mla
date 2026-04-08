import torch


# ===========================================================================
# Task 1: Dot Product
# ===========================================================================

def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product of two vectors a and b."""

    assert a.ndim == 1 and b.ndim == 1, "Input tensors must be 1D vectors."
    assert a.size() == b.size(), "Input vectors must have the same size."

    result = torch.tensor(0.0)

    for i in range(a.size(0)):
        result += a[i] * b[i]

    return result


# ===========================================================================
# Task 2: Matrix–Matrix Multiplication
# ===========================================================================

def matmul_loops(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product C = A @ B via nested for loops."""

    assert A.ndim == 2 and B.ndim == 2, "Input tensors must be 2D matrices."
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Incompatible matrix dimensions"
    
    C = torch.zeros(m, n)

    for i in range(m):
        for l in range(n):
            for k3 in range(k):
                C[i,l] += A[i,k3] * B[k3,l]

    return C


def matmul_dot(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Matrix product C = A @ B via slicing and calls to dot_product."""

    assert A.ndim == 2 and B.ndim == 2, "Input tensors must be 2D matrices."
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "Incompatible matrix dimensions"
    
    C = torch.zeros(m, n)
    
    for i in range(m):
        for l in range(n):
            C[i,l] = dot_product(A[i,:], B[:,l])

    return C


# ===========================================================================
# Task 3: Einsum  acsxp, bspy -> abcxy
# ===========================================================================

def einsum_loops(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Einsum acsxp, bspy -> abcxy via nested for loops."""

    assert A.ndim == 5 and B.ndim == 4, "Input tensors must have the correct number of dimensions."
    assert A.size() == torch.Size([2, 4, 5, 4, 3]), "Input tensor A must have shape [2, 4, 5, 4, 3]."
    assert B.size() == torch.Size([3, 5, 3, 5]), "Input tensor B must have shape [3, 5, 3, 5]."

    size_a, size_c, size_s, size_x, size_p = A.shape
    size_b, size_y = B.shape[0], B.shape[3]

    C = torch.zeros(size_a, size_b, size_c, size_x, size_y)
    # TODO: implement using for loops over all seven index dimensions

    return C


def einsum_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Einsum acsxp, bspy -> abcxy via loops over a, b, c, s and a GEMM (xp, py -> xy)."""

    assert A.ndim == 5 and B.ndim == 4, "Input tensors must have the correct number of dimensions."
    assert A.size() == torch.Size([2, 4, 5, 4, 3]), "Input tensor A must have shape [2, 4, 5, 4, 3]."
    assert B.size() == torch.Size([3, 5, 3, 5]), "Input tensor B must have shape [3, 5, 3, 5]."

    size_a, size_c, size_s, size_x, size_p = A.shape
    size_b, size_y = B.shape[0], B.shape[3]

    C = torch.zeros(size_a, size_b, size_c, size_x, size_y)
    # TODO: implement with for loops over a, b, c, s and a matmul for the inner GEMM

    return C


# ===========================================================================
# Task runners
# ===========================================================================

def task1():
    v1 = torch.rand(128)
    v2 = torch.rand(128)

    result_custom = dot_product(v1, v2)
    result_torch  = torch.dot(v1, v2)
    assert torch.allclose(result_custom, result_torch), (
        f"Task 1 mismatch: custom={result_custom:.6f}, torch={result_torch:.6f}"
    )
    print("Task 1 passed!")


def task2():
    A = torch.rand(8, 32)
    B = torch.rand(32, 16)

    result_loops = matmul_loops(A, B)
    result_dot   = matmul_dot(A, B)
    result_torch = torch.matmul(A, B)
    assert torch.allclose(result_loops, result_torch, atol=1e-5), (
        "Task 2 matmul_loops mismatch!"
    )
    assert torch.allclose(result_dot, result_torch, atol=1e-5), (
        "Task 2 matmul_dot mismatch!"
    )
    print("Task 2 passed!")


def task3():
    # A has shape [a, c, s, x, p] = [2, 4, 5, 4, 3]
    # B has shape [b, s, p, y]    = [3, 5, 3, 5]
    # C has shape [a, b, c, x, y] = [2, 3, 4, 4, 5]
    A_ein = torch.rand(2, 4, 5, 4, 3)
    B_ein = torch.rand(3, 5, 3, 5)

    reference    = torch.einsum("acsxp, bspy -> abcxy", A_ein, B_ein)
    result_loops = einsum_loops(A_ein, B_ein)
    result_gemm  = einsum_gemm(A_ein, B_ein)
    assert torch.allclose(result_loops, reference, atol=1e-5), (
        "Task 3A mismatch: pure-loop einsum differs from reference!"
    )
    assert torch.allclose(result_gemm, reference, atol=1e-5), (
        "Task 3B mismatch: loop+GEMM einsum differs from reference!"
    )
    print("Task 3 passed!")


def main():
    task1()
    task2()
    task3()


if __name__ == "__main__":
    main()
