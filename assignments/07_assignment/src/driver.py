"""
XRT Python driver for Assignment 07.

Usage (from the assignment directory, after building xclbins):
    python3 src/driver.py vadd
    python3 src/driver.py custom_vadd

Requires: pyxrt, numpy, torch
"""

import sys
import numpy as np
import torch
import pyxrt

# ── Buffer dimensions ─────────────────────────────────────────────────────────
VADD_N = 64  # number of BF16 elements


def verify(kernel: str, in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    """
    Verify the NPU output against a CPU reference.

    Parameters
    ----------
    kernel : "vadd" or "custom_vadd"
    in0, in1 : bfloat16 torch tensors storing raw BF16 bit patterns
    out : bfloat16 torch tensor
    """

    # TODO: implement verify() for both kernels.

    raise NotImplementedError("verify() not yet implemented")


def run(kernel_name: str) -> None:
    xclbin_path = f"build/final_{kernel_name}.xclbin"
    insts_path = f"build/insts_{kernel_name}.bin"

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

    data_in0 = torch.randn(VADD_N, dtype=torch.bfloat16)
    data_in1 = torch.randn(VADD_N, dtype=torch.bfloat16)
    data_out = torch.zeros(VADD_N, dtype=torch.bfloat16)

    # Create buffer objects with corresponding size
    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    # Copy data to buffer objects
    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    # View buffer objects as torch tensor
    tensor_in0 = torch.frombuffer(
        bo_in0.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in0.shape)
    ).view(data_in0.shape)
    tensor_in1 = torch.frombuffer(
        bo_in1.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in1.shape)
    ).view(data_in1.shape)
    tensor_out = torch.frombuffer(
        bo_out.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_out.shape)
    ).view(data_out.shape)
    assert torch.equal(data_in0, tensor_in0)
    assert torch.equal(data_in1, tensor_in1)
    assert torch.equal(data_out, tensor_out)

    # Sync buffer objects: to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output buffer object: from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    verify(kernel_name, tensor_in0, tensor_in1, tensor_out)

    print(f"[PASS] {kernel_name} verification passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("vadd", "custom_vadd"):
        print(f"Usage: {sys.argv[0]} <vadd|custom_vadd>")
        sys.exit(1)
    run(sys.argv[1])
