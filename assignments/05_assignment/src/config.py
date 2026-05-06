
from dataclasses import dataclass
import enum

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

"""
Requirements:
    Classify each dimension index automatically by inspecting in which tensors it appears.
    Compute strides for every tensor assuming row-major layout. A stride of 0 indicates that the dimension does not appear in that tensor.
    Set all exec_types to SEQ.
    Set data_type = DataType.FLOAT16, prim_main = PrimType.GEMM, prim_last = LastType.NONE, prim_first = FirstType.ZERO.
"""

import re

def generate_config(einsum: str, input_shapes: list[tuple[int]]) -> Config:
    # 3 catpure groups: 1 for the output, 2 for the inputs, ignore whitespaces
    einsum = re.sub(r'\s+', '', einsum)
    A_dims, B_dims, C_dims = re.match(r"([a-z]+),([a-z]+)->([a-z]+)", einsum).groups()
    
    # to keep the order of the dimensions as they appear in the einsum (and in the lecture)
    def remove_duplicates_keep_order(seq):
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]
    
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

    strides = []
    for tensor_dims in [A_dims, B_dims, C_dims]:
        stride = 1
        current = []
        for size, name in zip(dim_sizes[::-1], dim_names[::-1]):
            if name not in tensor_dims:
                current = [0] + current
            else:
                current = [stride] + current
                stride *= size
        strides.append(current)
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

if __name__ == "__main__":
    test_generate_config()