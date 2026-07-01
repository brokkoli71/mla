import numpy as np
import torch
import pyxrt


def run() -> None:
    xclbin_path = "build/final_matmul.xclbin"
    insts_path = "build/insts_matmul.bin"

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
    # value = 2 ^ (exponent - 127) * (mant*64)
    values = [1, 0.5, 0.25, -1, 2, 0.125, 0.03125, 0]
    # exponent 128 (derived from largest value; here 2)
    exponent = 128
    mant = [32, 16, 8, -32, 64, 4, 1, 0]
    expected = [exponent] + mant
    for v,m in zip(values, mant):
        assert v == pow(2,(exponent - 127)) * (m/64)

    # data_in0 = torch.zeros(16, 64, dtype=torch.bfloat16)
    # data_in1 = torch.zeros(64, 16, dtype=torch.bfloat16)

    size = 16*64
    data_in0 = torch.array(values*(size/8), dtype=torch.bfloat16).reshape(16, 64)
    data_in1 = torch.array(values*(size/8), dtype=torch.bfloat16).reshape(64, 16)

    expected_in0 = torch.array(expected*(size/8), dtype=torch.int8)
    expected_in1 = torch.array(expected*(size/8), dtype=torch.int8)
    data_out = torch.zeros(9*size/8, dtype=torch.int8)

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
    tensor_out0 = torch.frombuffer(
        bo_in0.map(),
        dtype=torch.int8,
        count=np.prod(data_in0.shape)
    ).view(data_in0.shape)
    assert torch.equal(data_in0, tensor_in0)
    assert torch.equal(data_in1, tensor_in1)

    # Sync buffer objects: to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output buffer object: from device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    assert torch.equal(data_out, expected_out)
    print("[PASS] matmul verification passed.")


if __name__ == "__main__":
    run()
