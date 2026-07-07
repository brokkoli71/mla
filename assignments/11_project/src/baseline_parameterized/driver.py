"""
XRT Python driver for the parameterized tensor kernel

    tensor_kernel_m1xn1xk1_bf16_bf16_bf16  (config m2=3, n2=1, k2=8)

Computation:  C = A @ B   with
    A : (M, K) = (48, 64) bf16
    B : (K, N) = (64, 16) bf16   (passed to the kernel transposed, N x K)
    C : (M, N) = (48, 16) bf16

Tile factors baked into the .mlir:
    m2 = 3, n2 = 1, k2 = 8,  sub-tiles 2x2, micro-tile 8x8
      M = m2 * 2 * 8 = 48
      N = n2 * 2 * 8 = 16
      K = k2 * 8     = 64

Per the kernel's einsum (from its .s header):
    OUT[m2 n2 m1 n1 m0 n0] += IN0[m2 k1 m1 m0 k0] * IN1[n2 k1 n1 k0 n0]

L1 tiled layouts the DMA delivers verbatim (host must pre-tile):
    in0 : [m2][k1][m1][m0][k0] = <3x8x2x8x8>   (A,  natural M x K)
    in1 : [n2][k1][n1][k0][n0] = <1x8x2x8x8>   (B,  natural K x N; note k0 before n0)
    out : [m2][n2][m1][n1][m0][n0] = <3x1x2x2x8x8>

Usage (from the assignment directory, after building the xclbin):
    python3 src/baseline_parameterized/driver.py

Requires: pyxrt, numpy, torch
"""

import os

import numpy as np
import torch
import pyxrt

# ── problem / tiling constants ────────────────────────────────────────────────
# Override via env (must match the built .mlir instance): TK_M2 / TK_N2 / TK_K2.
M2 = int(os.environ.get("TK_M2", 3))     # outer M tiles
N2 = int(os.environ.get("TK_N2", 1))     # outer N tiles
K2 = int(os.environ.get("TK_K2", 8))     # k1 (K/8)
MSUB, NSUB = 2, 2          # sub-tiles per output block (fixed |m1|=|n1|=2)
T = 8                      # micro-tile edge (fixed |m0|=|n0|=|k0|=8)

M = M2 * MSUB * T          # M = m2 * 16
N = N2 * NSUB * T          # N = n2 * 16
K = K2 * T                 # K = k1 * 8


def tile_a(a: torch.Tensor) -> torch.Tensor:
    """(M, K) row-major -> [m2][k2][m_sub][m][k] = <3x8x2x8x8>."""
    #   row = m2*(MSUB*T) + m_sub*T + m ,  col = k2*T + k
    return (
        a.reshape(M2, MSUB, T, K2, T)   # [m2][m_sub][m][k2][k]
        .permute(0, 3, 1, 2, 4)         # [m2][k2][m_sub][m][k]
        .contiguous()
    )


def tile_b(b: torch.Tensor) -> torch.Tensor:
    """(K, N) row-major -> [n2][k1][n1][k0][n0] = <1x8x2x8x8>."""
    #   IN1[n2,k1,n1,k0,n0] = B[k1*T + k0][n2*(NSUB*T) + n1*T + n0]
    return (
        b.reshape(K2, T, N2, NSUB, T)   # [k1][k0][n2][n1][n0]
        .permute(2, 0, 3, 1, 4)         # [n2][k1][n1][k0][n0]
        .contiguous()
    )


def untile_c(out: torch.Tensor) -> torch.Tensor:
    """[m2][n2][m_sub][n_sub][m][n] = <3x1x2x2x8x8> -> (M, N) row-major."""
    #   row = m2*(MSUB*T) + m_sub*T + m ,  col = n2*(NSUB*T) + n_sub*T + n
    return (
        out.reshape(M2, N2, MSUB, NSUB, T, T)
        .permute(0, 2, 4, 1, 3, 5)      # [m2][m_sub][m][n2][n_sub][n]
        .reshape(M, N)
        .contiguous()
    )


def verify(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Verify the NPU output (row-major) against a CPU reference C = A @ B."""
    ref = a.float() @ b.float()
    torch.testing.assert_close(out.float(), ref, atol=0.5, rtol=0.02)


def run() -> None:
    xclbin_path = os.environ.get(
        "TK_XCLBIN", "build/final_tensor_kernel_param.xclbin"
    )
    insts_path = os.environ.get(
        "TK_INSTS", "build/insts_tensor_kernel_param.bin"
    )

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
    A = torch.randn(M, K, dtype=torch.bfloat16)     # (48, 64)
    B = torch.randn(K, N, dtype=torch.bfloat16)     # (64, 16), natural K x N

    data_in0 = tile_a(A)                             # <3x8x2x8x8>
    data_in1 = tile_b(B)                             # <1x8x2x8x8>  (B tiled, K x N)
    data_out = torch.zeros(M2, N2, MSUB, NSUB, T, T, dtype=torch.bfloat16)

    # Create buffer objects with corresponding size
    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    # Copy data to buffer objects
    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    # View output buffer as torch tensor
    tensor_out = torch.frombuffer(
        bo_out.map(),
        dtype=torch.bfloat16,
        count=int(np.prod(data_out.shape)),
    ).view(data_out.shape)

    # Sync inputs to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output back
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    out_rowmajor = untile_c(tensor_out)
    verify(A, B, out_rowmajor)

    print(
        f"[PASS] tensor_kernel_m1xn1xk1 (m2={M2},n2={N2},k1={K2}; "
        f"{M}x{K} @ {K}x{N}) verification passed."
    )


if __name__ == "__main__":
    run()
