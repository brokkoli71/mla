"""
Software simulation test for the BF16->BFP16 conversion in conv().

conv() converts both input matrices in-place:
  BF16 in L1 -> (intermediate FP32) -> BFP16ebs8 in the same L1 memory

BFP16ebs8 format (XDNA2):
  - 8 elements per block (ebs = exponent block size 8)
  - 1 byte shared exponent per block (biased IEEE-754 style)
  - 1 byte signed mantissa per element (two's complement)
  -> 9 bytes per 8-element block

This file tests the quantization behaviour in software so we know what
precision to expect from the hardware output once matmul is implemented.
Run with:  python3 src/size16/test_conv.py
"""

import numpy as np
import torch

BLOCK_SIZE = 8  # elements per BFP16 block


def bfp16ebs8_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate BFP16ebs8: every BLOCK_SIZE elements share one 8-bit biased exponent.
    Each element gets an 8-bit signed mantissa (two's complement).

    Encoding (from driver_verify_conversion.py assertion):
        value = 2^(biased_exp - 127) * (mantissa / 64)
      where biased_exp = floor(log2(max_abs)) + 127

    Equivalently:
        scale     = 2^(floor(log2(max_abs)) - 6)
        mantissa  = round(value / scale)   clamped to [-128, 127]
        recovered = mantissa * scale
    """
    flat = x.float().reshape(-1)
    result = torch.zeros_like(flat)

    for i in range(0, len(flat), BLOCK_SIZE):
        block = flat[i : i + BLOCK_SIZE]
        max_abs = block.abs().max().item()
        if max_abs == 0.0:
            continue

        shared_exp = int(np.floor(np.log2(max_abs)))
        scale = 2.0 ** (shared_exp - 6)   # max element gives mantissa ~64

        mantissas = (block / scale).round().clamp(-128, 127).int()
        result[i : i + BLOCK_SIZE] = mantissas.float() * scale

    return result.reshape(x.shape).to(x.dtype)


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def test_quantization_error():
    """
    Per-element relative error must stay within mantissa resolution.
    8-bit mantissa -> 1/128 ~ 0.78 % max relative error per element.
    We allow up to 2 % to cover edge cases at block boundaries.
    """
    torch.manual_seed(42)
    A = torch.randn(16, 64, dtype=torch.bfloat16)
    B = torch.randn(64, 16, dtype=torch.bfloat16)

    for name, mat in [("A (in0)", A), ("B (in1)", B)]:
        q = bfp16ebs8_quantize(mat)
        err = (mat.float() - q.float()).abs()
        rel = err / mat.float().abs().clamp(min=1e-6)
        max_rel = rel.max().item()

        assert max_rel < 0.02, f"{name}: relative error {max_rel:.4%} exceeds 2 %"
        print(f"  {name}: max relative error = {max_rel:.4%}  OK")


def test_zero_block():
    """A block of all zeros must encode and decode back to zero."""
    A = torch.zeros(16, 64, dtype=torch.bfloat16)
    q = bfp16ebs8_quantize(A)
    assert q.abs().max().item() == 0.0, "Zero block did not round-trip to zero"
    print("  Zero block: OK")


def test_sign_preservation():
    """Signs must be preserved for every non-tiny element."""
    torch.manual_seed(7)
    A = torch.randn(16, 64, dtype=torch.bfloat16)
    q = bfp16ebs8_quantize(A)

    mask = A.float().abs() > 1e-4
    mismatch = (torch.sign(A.float()[mask]) != torch.sign(q.float()[mask])).sum()
    assert mismatch == 0, f"{mismatch} sign errors after BFP16 quantization"
    print("  Sign preservation: OK")


def test_matmul_error_budget():
    """
    Show the expected error introduced by BFP16 quantization in the full matmul.

    When matmul is implemented, the NPU output should match:
        bfp16ebs8_quantize(A) @ bfp16ebs8_quantize(B)
    rather than the plain A @ B.
    This test shows how large that difference is so we can set atol/rtol correctly
    in driver.py.
    """
    torch.manual_seed(42)
    A = torch.randn(16, 64, dtype=torch.bfloat16)
    B = torch.randn(64, 16, dtype=torch.bfloat16)

    naive   = A.float() @ B.float()
    quant   = bfp16ebs8_quantize(A).float() @ bfp16ebs8_quantize(B).float()

    abs_diff = (naive - quant).abs()
    rel_diff = abs_diff / naive.abs().clamp(min=1e-4)

    print(f"  Matmul abs error (naive vs BFP16-quantized): "
          f"max={abs_diff.max():.4f}  mean={abs_diff.mean():.4f}")
    print(f"  Matmul rel error: max={rel_diff.max():.4%}  mean={rel_diff.mean():.4%}")
    print(f"  -> Use atol ~ {abs_diff.max():.2f}, rtol ~ {rel_diff.max():.2f} in driver verify()")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BFP16ebs8 conversion simulation")
    print("=" * 48)

    print("\n[1] Per-element quantization error")
    test_quantization_error()

    print("\n[2] Zero block round-trip")
    test_zero_block()

    print("\n[3] Sign preservation")
    test_sign_preservation()

    print("\n[4] Expected matmul error budget (sets driver tolerances)")
    test_matmul_error_budget()

    print("\n[ALL PASSED]")
