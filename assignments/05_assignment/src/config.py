
from dataclasses import dataclass
import enum
from typing import Self

"""
- **`DimType`**: `M`, `N`, `K`, `C`
- **`ExecType`**: `SEQ`, `PAR`, `PRIM`
- **`PrimType`**: `GEMM`, `BGEMM`
- **`LastType`**: `NONE`, `ELWISE_MUL`
- **`FirstType`**: `ZERO`
- **`DataType`**: `FLOAT16`, `FLOAT32`
"""

class DimType(enum.Enum):
    M = 0
    N = 1
    K = 2
    C = 3

class ExecType(enum.Enum):
    SEQ = 0
    PAR = 1
    PRIM = 2

class PrimType(enum.Enum):
    GEMM = 0
    BGEMM = 1

class LastType(enum.Enum):
    NONE = 0
    ELWISE_MUL = 1

class FirstType(enum.Enum):
    ZERO = 0

class DataType(enum.Enum):
    FLOAT16 = 0
    FLOAT32 = 1

"""
Field 	Type 	Description
data_type 	DataType 	Numeric precision of the operands
prim_main 	PrimType 	Main (B)GEMM primitive used inside the kernel
prim_last 	LastType 	Optional elementwise operation applied after the accumulation
prim_first 	FirstType 	Initialization of the accumulator
dim_types 	list[DimType] 	Per-dimension index type
exec_types 	list[ExecType] 	Per-dimension execution strategy
dim_sizes 	list[int] 	Per-dimension size
strides 	list[list[int]] 	Per-tensor, per-dimension stride (one inner list per tensor)
"""
@dataclass
class Config():
    data_type: DataType
    prim_main: PrimType
    dim_types: list[DimType]
    exec_types: list[ExecType]
    dim_sizes: list[int]
    strides: list[list[int]]
    prim_last: LastType = LastType.NONE
    prim_first: FirstType = FirstType.ZERO

    def from_config(config: Self, **kwargs):
        return Config(
            data_type=kwargs.get("data_type", config.data_type),
            prim_main=kwargs.get("prim_main", config.prim_main),
            dim_types=kwargs.get("dim_types", config.dim_types),
            exec_types=kwargs.get("exec_types", config.exec_types),
            dim_sizes=kwargs.get("dim_sizes", config.dim_sizes),
            strides=kwargs.get("strides", config.strides),
            prim_last=kwargs.get("prim_last", config.prim_last),
            prim_first=kwargs.get("prim_first", config.prim_first)
        )

    def __str__(self):
        return f"""Config(
    data_type={self.data_type},
    prim_main={self.prim_main},
    prim_last={self.prim_last},
    prim_first={self.prim_first},
    dim_types={self.dim_types},
    exec_types={self.exec_types},
    dim_sizes={self.dim_sizes},
    strides={self.strides}
)"""

import re

def generate_config(einsum: str, input_shapes: list[tuple[int]], dim_order: str | None = None) -> Config:
    # 3 catpure groups: 1 for the output, 2 for the inputs, ignore whitespaces
    einsum = re.sub(r'\s+', '', einsum)
    A_dims, B_dims, C_dims = re.match(r"([a-z]+),([a-z]+)->([a-z]+)", einsum).groups()
    
    # to keep the order of the dimensions as they appear in the einsum (and in the lecture)
    def remove_duplicates_keep_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
    if dim_order is not None:
        dim_names = list(dim_order)
    else:
        dim_names = remove_duplicates_keep_order(A_dims + B_dims + C_dims)

    dim_types = []
    dim_sizes = []
    for dim in dim_names:
        if dim in A_dims and dim in B_dims and dim in C_dims:
            dim_type = DimType.C
        elif dim in A_dims and dim in C_dims and not dim in B_dims:
            dim_type = DimType.M
        elif dim in B_dims and dim in C_dims and not dim in A_dims:
            dim_type = DimType.N
        elif dim in A_dims and dim in B_dims and not dim in C_dims:
            dim_type = DimType.K
        else:
            raise ValueError(f"Dimension {dim} does not fit into M, N, K, C categories.")
        dim_types.append(dim_type)

        # Determine the size of the dimension from the input shapes
        if dim in A_dims:
            size = input_shapes[0][A_dims.index(dim)]
        elif dim in B_dims:
            size = input_shapes[1][B_dims.index(dim)]
        else:  
            raise ValueError(f"Dimension {dim} not found in any input tensor.")
        dim_sizes.append(size)

    exec_types = [ExecType.SEQ] * len(dim_types)

    size_of = dict(zip(dim_names, dim_sizes))

    strides = []
    for tensor_dims in [A_dims, B_dims, C_dims]:
        own_strides = {}
        stride = 1
        for name in tensor_dims[::-1]:
            own_strides[name] = stride
            stride *= size_of[name]
        # Map onto the global dimension slots; 0 means "not in this tensor".
        strides.append([own_strides.get(name, 0) for name in dim_names])
    return Config(
        data_type=DataType.FLOAT16,
        prim_main=PrimType.GEMM,
        dim_types=dim_types,
        exec_types=exec_types,
        dim_sizes=dim_sizes,
        strides=strides
    )

def test_generate_config():
    einsum = "wvxy, wvyz -> wxz"
    input_shapes = [(4, 2, 4096, 4096), (4, 2, 4096, 4096)]
    config = generate_config(einsum, input_shapes)
    print(config)

def test_generate_config_consistent_dim_order():
    config = generate_config("cmk, ckn -> cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    assert config.dim_sizes == [4, 4096, 4096, 4096], config.dim_sizes
    assert config.strides == [
        [16777216, 4096, 1, 0],       # cmk
        [16777216, 0, 4096, 1],       # ckn
        [16777216, 4096, 0, 1],       # cmn
    ], config.strides
    print("test_generate_config_consistent_dim_order passed!")

def test_generate_config_permuted_dim_order():
    def row_major(shape):
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return strides

    # n comes before k in B, but after k in the global order m, k, n.
    config = generate_config("mk, nk -> mn", [(32, 8), (16, 8)])
    assert config.dim_sizes == [32, 8, 16], config.dim_sizes
    m_a, k_a = row_major((32, 8))
    n_b, k_b = row_major((16, 8))
    m_c, n_c = row_major((32, 16))
    assert config.strides == [
        [m_a, k_a, 0],                # mk
        [0, k_b, n_b],                # nk  -> k is the contiguous axis, not n
        [m_c, 0, n_c],                # mn
    ], config.strides

    a, c, s, p, x, b, y = 4, 3, 64, 64, 1536, 4, 1152
    config = generate_config("acspx, bspy -> abcyx", [(a, c, s, p, x), (b, s, p, y)])
    assert config.dim_sizes == [a, c, s, p, x, b, y], config.dim_sizes
    a_a, c_a, s_a, p_a, x_a = row_major((a, c, s, p, x))
    b_b, s_b, p_b, y_b = row_major((b, s, p, y))
    a_c, b_c, c_c, y_c, x_c = row_major((a, b, c, y, x))
    assert config.strides == [
        [a_a, c_a, s_a, p_a, x_a, 0, 0],      # acspx
        [0, 0, s_b, p_b, 0, b_b, y_b],        # bspy
        [a_c, c_c, 0, 0, x_c, b_c, y_c],      # abcyx
    ], config.strides
    print("test_generate_config_permuted_dim_order passed!")

if __name__ == "__main__":
    test_generate_config()
    test_generate_config_consistent_dim_order()
    test_generate_config_permuted_dim_order()