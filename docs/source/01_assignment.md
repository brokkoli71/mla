# Assignment 01: Tensors and Einsum

In this assignment you will implement several tensor operations from scratch using Python and PyTorch tensors. All code should be written in `src/assignment_01.py`. Starter code with vector/tensor initializations, function headers, and assertion-based tests is already provided. **Please do not modify the test code**.

---

## Task 1: Dot Product

The *dot product* of two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ is defined as

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{k=0}^{n-1} a_k \cdot b_k
$$

**Your task** is to implement the function `dot_product(a, b)` in `src/assignment_01.py`. Use a python `for` loop to iterate over the elements of the input vectors. The function should be written in a way that it can handle vectors of any length (not just the provided length of 100).

---

## Task 2: Matrix-Matrix Multiplication

The matrix product $C = A \cdot B$ for $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$ and $C \in \mathbb{R}^{m \times n}$ is defined element-wise as

$$
c_{pr} = \sum_{q=0}^{k-1} a_{pq} \cdot b_{qr}
$$

This corresponds to the einsum `pq, qr -> pr`.

**Your task** consists of two parts:

1. **Implement the function** `matmul_loops(A, B)`: Compute $C = A \cdot B$ using nested `for` loops. No PyTorch matrix-multiply operations are allowed inside this function.
2. **Implement the function** `matmul_dot(A, B)`: Compute the same product but reuse your `dot_product` function from Task 1. To do so, use loops over the `dot_product` function that uses slices of the input matrices as 1-D views.

---

## Task 3: Einsum `acsxp, bspy -> abcxy`

Given tensors

$$
A \in \mathbb{R}^{a \times c \times s \times x \times p}, \quad
B \in \mathbb{R}^{b \times s \times p \times y}
$$

the target einsum `acsxp, bspy -> abcxy` computes

$$
C_{abcxy} = \sum_{s} \sum_{p} A_{acsxp} \cdot B_{bspy}
$$

You can assume a static shape for the tensors, `A: (2, 4, 5, 4, 3)`, `B: (3, 5, 3, 5)` and `C: (2, 3, 4, 4, 5)`

**Your task** consists of two parts:

1. **Implement the function** `einsum_loops(A, B)`: Compute the einsum expression by using `for` loops to iterate over all index dimensions, accumulating the products into a pre-allocated output tensor.
2. **Implement the function** `einsum_gemm(A, B)`: Compute the same einsum but reuse one of your matrix multiplication implementations from Task 2. To do so, use loops over the matrix multiplication function that uses slices of the input tensors as 2-D views.

---

## Optional Task: Tensor Permutation and Reshaping

Look up `torch.permute` and `torch.reshape` in the PyTorch documentation and experiment with them inside a new python file. Try to answer the following questions:

* How do `.shape` and `.stride()` change after a `torch.permute` or `torch.reshape` call? Does the underlying data move in memory?
* How does `torch.reshape` differ from `torch.view`?
* Why can reshaping a permuted tensor be handled differently than reshaping a freshly created tensor? (Hint: check `.is_contiguous()` before and after `torch.permute`).