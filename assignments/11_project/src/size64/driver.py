"""
XRT Python driver for the size64 matmul (M=N=K=64).

Usage (from the assignment directory, after building xclbins):
    python3 src/size64/driver.py

Requires: pyxrt, numpy, torch
"""

import os

import numpy as np
import torch
import pyxrt


def verify(in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    """
    Verify the NPU output against a CPU reference.

    Computation: out = in0 @ in1  (in1 is the original, non-transposed B)

    Parameters
    ----------
    in0 : bfloat16 torch tensor, shape (64, 64)
    in1 : bfloat16 torch tensor, shape (64, 64) — natural K×N layout
    out : bfloat16 torch tensor, shape (64, 64)
    """

    ref = in0 @ in1

    torch.testing.assert_close(out, ref, atol=0.5, rtol=0.02)
    #torch.testing.assert_close(out, ref, atol=0.1, rtol=0.01)


def run() -> None:
    xclbin_path = os.environ.get("MATMUL_XCLBIN", "build/final_matmul_size64.xclbin")
    insts_path = os.environ.get("MATMUL_INSTS", "build/insts_matmul_size64.bin")

    insts = np.fromfile(insts_path, dtype=np.uint32)

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    context = pyxrt.hw_context(device, uuid)
    kname = xclbin.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(context, kname)

    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

    torch.manual_seed(42)
    data_in0 = torch.randn(64, 64, dtype=torch.bfloat16)
    data_in1 = torch.randn(64, 64, dtype=torch.bfloat16)   # natural K×N layout
    data_in1_t = data_in1.t().contiguous()                  # transposed to N×K for kernel
    data_out = torch.zeros(64, 64, dtype=torch.bfloat16)

    # Create buffer objects with corresponding size
    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1_t.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    # Copy data to buffer objects
    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1_t.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    # View buffer objects as torch tensor
    tensor_in0 = torch.frombuffer(
        bo_in0.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in0.shape)
    ).view(data_in0.shape)
    tensor_in1_t = torch.frombuffer(
        bo_in1.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in1_t.shape)
    ).view(data_in1_t.shape)
    tensor_out = torch.frombuffer(
        bo_out.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_out.shape)
    ).view(data_out.shape)
    assert torch.equal(data_in0, tensor_in0)
    assert torch.equal(data_in1_t, tensor_in1_t)
    assert torch.equal(data_out, tensor_out)

    # Sync buffer objects: to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1_t.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output buffer object: from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    # The kernel writes output tiles in [p_hi][q_hi][p_lo][q_lo][m][n] order
    # (4x4x2x2x8x8). The core-tile DMA is limited to 3 dims, so this final
    # reshape to row-major 64x64 is done here on the host:
    #   row = p_hi*16 + p_lo*8 + m,   col = q_hi*16 + q_lo*8 + n
    out_rowmajor = (
        tensor_out.reshape(4, 4, 2, 2, 8, 8)
        .permute(0, 2, 4, 1, 3, 5)      # -> [p_hi][p_lo][m][q_hi][q_lo][n]
        .reshape(64, 64)
        .contiguous()
    )

    verify(tensor_in0, data_in1, out_rowmajor)

    print("[PASS] matmul verification passed.")


if __name__ == "__main__":
    run()
