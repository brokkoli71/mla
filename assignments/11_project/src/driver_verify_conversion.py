"""
Hardware test for the BF16->BFP16 conversion stage (conv kernel).

Tests that the full pipeline  conv() + matmul()  produces a result consistent
with BFP16-quantised arithmetic rather than plain BF16 arithmetic.

The reference is:
    bfp16ebs8_quantize(A) @ bfp16ebs8_quantize(B)

because conv() converts A and B to BFP16 in-place before matmul() uses them.

NOTE: matmul() is currently a stub — the test will FAIL with "output is all
zeros" until the vmac body is implemented.  The expected tensor printed on
failure shows exactly what the correct output should look like.

Run via:
    make run_verify_bfp16
"""

import numpy as np
import torch
import pyxrt


# ---------------------------------------------------------------------------
# BFP16ebs8 simulation  (mirrors the hardware conv() kernel)
# ---------------------------------------------------------------------------

_BLOCK = 8  # elements per BFP16 block


def bfp16ebs8_quantize(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate the BFP16ebs8 quantization performed by conv().

    Format: 1 shared-exp byte + 8 signed-mantissa bytes per 8 elements.
        value = 2^(biased_exp - 127) * (mantissa / 64)
        biased_exp = floor(log2(max_abs)) + 127
        scale      = 2^(floor(log2(max_abs)) - 6)
    """
    flat = x.float().reshape(-1)
    result = torch.zeros_like(flat)

    for i in range(0, len(flat), _BLOCK):
        block = flat[i : i + _BLOCK]
        max_abs = block.abs().max().item()
        if max_abs == 0.0:
            continue
        shared_exp = int(np.floor(np.log2(max_abs)))
        scale = 2.0 ** (shared_exp - 6)
        mantissas = (block / scale).round().clamp(-128, 127).int()
        result[i : i + _BLOCK] = mantissas.float() * scale

    return result.reshape(x.shape).to(x.dtype)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    """
    Parameters
    ----------
    in0 : (16, 64) bfloat16  — matrix A, original values
    in1 : (64, 16) bfloat16  — matrix B, original K×N layout (not transposed)
    out : (16, 16) bfloat16  — NPU output
    """
    ref_bfp16 = bfp16ebs8_quantize(in0).float() @ bfp16ebs8_quantize(in1).float()
    ref_plain  = in0.float() @ in1.float()

    print("\n--- Debug ---")
    print(f"NPU output range:   [{out.float().min():.3f}, {out.float().max():.3f}]")
    print(f"BFP16 ref range:    [{ref_bfp16.min():.3f}, {ref_bfp16.max():.3f}]")
    print(f"Plain A@B range:    [{ref_plain.min():.3f}, {ref_plain.max():.3f}]")

    diff = (out.float() - ref_bfp16).abs()
    zeros = (out.float() == 0).sum().item()
    print(f"Zero elements in output: {zeros}/256")
    print(f"Max abs error vs BFP16 ref: {diff.max():.3f}")
    print(f"Mean abs error vs BFP16 ref: {diff.mean():.3f}")

    diff_plain = (out.float() - ref_plain).abs()
    print(f"Max abs error vs plain A@B: {diff_plain.max():.3f}")

    # Check if output looks like a permuted version of the reference
    print(f"\nNPU out (first 4×4):\n{out[:4,:4].float()}")
    print(f"BFP16 ref (first 4×4):\n{ref_bfp16[:4,:4]}")
    print(f"Plain ref (first 4×4):\n{ref_plain[:4,:4]}")
    print("--- End debug ---\n")

    ref = ref_bfp16.to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=4.0, rtol=0.05)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run() -> None:
    xclbin_path = "build/final_matmul_size16.xclbin"
    insts_path  = "build/insts_matmul_size16.bin"

    insts = np.fromfile(insts_path, dtype=np.uint32)

    device  = pyxrt.device(0)
    xclbin  = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid    = xclbin.get_uuid()
    context = pyxrt.hw_context(device, uuid)
    kname   = xclbin.get_kernels()[0].get_name()
    kernel  = pyxrt.kernel(context, kname)

    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

    torch.manual_seed(42)
    data_in0   = torch.randn(16, 64, dtype=torch.bfloat16)
    data_in1   = torch.randn(64, 16, dtype=torch.bfloat16)  # natural K×N
    data_in1_t = data_in1.t().contiguous()                   # N×K for kernel
    data_out   = torch.zeros(16, 16, dtype=torch.bfloat16)

    bo_in0 = pyxrt.bo(device, data_in0.nbytes,   pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1_t.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes,   pyxrt.bo.host_only, 0)

    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1_t.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    tensor_in0 = torch.frombuffer(bo_in0.map(), dtype=torch.bfloat16,
                                  count=data_in0.numel()).view(data_in0.shape)
    tensor_out = torch.frombuffer(bo_out.map(), dtype=torch.bfloat16,
                                    count=data_out.numel()).view(data_out.shape)

    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes,   0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1_t.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes,   0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    verify(tensor_in0, data_in1, tensor_out)

    print("[PASS] BFP16 conversion + matmul verification passed.")


if __name__ == "__main__":
    run()
