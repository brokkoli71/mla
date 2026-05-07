from collections.abc import Set

from config import Config, DataType, PrimType, DimType, ExecType, generate_config

class Optimizer:
    
    def __init__(self, config: Config):
        self.config = config

    """Returns a new configuration with the specified dimension split into two. The optimizer's current configuration is replaced."""
    def split_dim(self, dim_id: int, outer_size: int, inner_size: int):
        original_size = self.config.dim_sizes[dim_id]

        if outer_size * inner_size != original_size:
            raise ValueError(f"Outer size {outer_size} and inner size {inner_size} do not multiply to original size {original_size}")
        
        new_config = Config(
            data_type=self.config.data_type,
            prim_main=self.config.prim_main,
            dim_types=self.config.dim_types[:dim_id] + [self.config.dim_types[dim_id]] * 2 + self.config.dim_types[dim_id+1:],
            exec_types=self.config.exec_types[:dim_id] + [self.config.exec_types[dim_id]] * 2 + self.config.exec_types[dim_id+1:],
            dim_sizes=self.config.dim_sizes[:dim_id] + [outer_size, inner_size] + self.config.dim_sizes[dim_id+1:],
            # strides=self.config.strides[:dim_id] + [[self.config.strides[0][dim_id] * inner_size, self.config.strides[0][dim_id]]] + self.config.strides[dim_id+1:],
            strides=[stride[:dim_id] + [stride[dim_id] * inner_size, stride[dim_id]] + stride[dim_id+1:] for stride in self.config.strides],
            prim_last=self.config.prim_last,
            prim_first=self.config.prim_first
        )
        self.config = new_config
        return new_config

    """Returns a new configuration with the specified dimensions fused into one. The optimizer's current configuration is replaced."""
    def fuse_dims(self, dim_id_a: int, dim_id_b: int):
        # set dim_id_a to be the smaller one to simplify the logic
        if dim_id_a > dim_id_b:
            dim_id_a, dim_id_b = dim_id_b, dim_id_a

        if self.config.dim_types[dim_id_a] != self.config.dim_types[dim_id_b]:
            raise ValueError(f"Dimensions {dim_id_a} and {dim_id_b} have different types and cannot be fused.")
        for i, stride in enumerate(self.config.strides):
            if stride[dim_id_a] == 0: # if both have same type and one of them is 0, the other must also be 0
                continue
            if not stride[dim_id_a] == stride[dim_id_b] * self.config.dim_sizes[dim_id_b]:
                raise ValueError(f"Dimensions {dim_id_a} and {dim_id_b} are not contiguous in dim {i} and cannot be fused.")

        new_dim_sizes = list(self.config.dim_sizes)
        new_dim_sizes[dim_id_a] = self.config.dim_sizes[dim_id_a] * self.config.dim_sizes[dim_id_b]
        del new_dim_sizes[dim_id_b]

        def new_stride(stride):
            new_stride = list(stride)
            new_stride[dim_id_a] = stride[dim_id_b] # stride of the fused dimension is the same as the second dimension
            del new_stride[dim_id_b]
            return new_stride
        
        
        new_config = Config(
            data_type=self.config.data_type,
            prim_main=self.config.prim_main,
            dim_types=self.config.dim_types[:dim_id_b] + self.config.dim_types[dim_id_b+1:], # keeps dim_type of a
            exec_types=self.config.exec_types[:dim_id_b] + self.config.exec_types[dim_id_b+1:], # keeps exec_type of a
            dim_sizes=new_dim_sizes,
            strides=[new_stride(stride) for stride in self.config.strides],
            prim_last=self.config.prim_last,
            prim_first=self.config.prim_first
        )
        self.config = new_config
        return new_config
    
    def permute_dims(self, permutation: list[int]):
        if sorted(permutation) != list(range(len(self.config.dim_types))):
            raise ValueError(f"New order {permutation} is not a valid permutation of dimensions.")
        new_config = Config(
            data_type=self.config.data_type,
            prim_main=self.config.prim_main,
            dim_types=[self.config.dim_types[i] for i in permutation],
            exec_types=[self.config.exec_types[i] for i in permutation],
            dim_sizes=[self.config.dim_sizes[i] for i in permutation],
            strides=[ [stride[i] for i in permutation] for stride in self.config.strides],
            prim_last=self.config.prim_last,
            prim_first=self.config.prim_first
        )
        self.config = new_config
        return new_config


    #d) Implement the function make_executable().
    #Set exec types and permute the config’s dimensions so that the config becomes executable via cuTile. Use the parallel execution type where possible. Test the resulting configuration with your verify() function from e).
    def make_executable(self):
        ...
    
    def verify(self):
        # No K-dimension may have exec_type = PAR.
        for dim_type, exec_type in zip(self.config.dim_types, self.config.exec_types):
            if dim_type == DimType.K and exec_type == ExecType.PAR:
                raise ValueError("K-dimension cannot have parallel execution type.")
        # order: PAR -> SEQ -> PRIM
        order = {ExecType.PAR: 0, ExecType.SEQ: 1, ExecType.PRIM: 2}
        sorted_exec_types = sorted(self.config.exec_types, key=lambda x: order[x])
        if self.config.exec_types != sorted_exec_types:
            raise ValueError("Execution types must be in order: PAR -> SEQ -> PRIM.")
        # The rightmost dimension must be PRIM and the PRIM dimensions must include at least one dimension of each type M, N, and K.
        prim_dim_types = set(dim_type for dim_type, exec_type in zip(self.config.dim_types, self.config.exec_types) if exec_type == ExecType.PRIM)
        if not prim_dim_types.issuperset({DimType.M, DimType.N, DimType.K}):
            raise ValueError("The rightmost dimensions must be PRIM and the PRIM dimensions must include at least one dimension of each type M, N, and K.")
        

def test_split_dim():
    input_shapes = [(4, 8), (8, 16)]
    einsum = "ab, bc -> ac"
    config = generate_config(einsum, input_shapes)
    optimizer = Optimizer(config)
    new_config = optimizer.split_dim(dim_id=1, outer_size=4, inner_size=2)
    expected_dim_sizes = [4, 4, 2, 16]
    assert new_config.dim_sizes == expected_dim_sizes, f"Expected dim sizes {expected_dim_sizes}, but got {new_config.dim_sizes}"
    expected_strides = [[8, 2, 1, 0], [0, 32, 16, 1], [16, 0, 0, 1]]
    assert new_config.strides == expected_strides, f"Expected strides {expected_strides}, but got {new_config.strides}"
    print("test_split_dim passed!")

def test_fuse_dims():
    input_shapes = [(2, 4, 8), (8, 16)]
    einsum = "klb, bc -> klc"
    config = generate_config(einsum, input_shapes)
    optimizer = Optimizer(config)
    fused_config = optimizer.fuse_dims(dim_id_a=0, dim_id_b=1)
    expected_dim_sizes = [8, 8, 16]
    assert fused_config.dim_sizes == expected_dim_sizes, f"Expected dim sizes {expected_dim_sizes}, but got {fused_config.dim_sizes}"
    expected_strides = [[8, 1, 0], [0, 16, 1], [16, 0, 1]]
    assert fused_config.strides == expected_strides, f"Expected strides {expected_strides}, but got {fused_config.strides}"
    print("test_fuse_dims passed!")

def test_split_fuse_dims():
    input_shapes = [(2, 4, 8), (8, 16)]
    einsum = "klb, bc -> klc"
    config = generate_config(einsum, input_shapes)
    optimizer = Optimizer(config)
    new_config = optimizer.split_dim(dim_id=0, outer_size=1, inner_size=2)
    fused_config = optimizer.fuse_dims(dim_id_a=0, dim_id_b=1)
    assert config == fused_config, f"Expected config to be the same after split and fuse, but got {fused_config}"
    print("test_split_fuse_dims passed!")

def test_permute_dims():
    # skip this since the logic is straightforward
    pass 
    
def test_verify():
    input_shapes = [(2, 2, 2, 2), (2, 2, 2, 2)]
    einsum = "abxy, bcyz -> acxz"
    dim_order = "abcxyz"
    config = generate_config(einsum, input_shapes, dim_order=dim_order)

    valid_exec_types = [ExecType.PAR, ExecType.SEQ, ExecType.SEQ, ExecType.PRIM, ExecType.PRIM, ExecType.PRIM]
    config.exec_types = valid_exec_types
    optimizer = Optimizer(config)
    optimizer.verify()

    config.exec_types[1] = ExecType.PAR
    expected_error_message = "K-dimension cannot have parallel execution type."
    try:
        optimizer.verify()
        assert False, "Expected ValueError for K-dimension with parallel execution type, but no error was raised."
    except ValueError as e:
        assert str(e) == expected_error_message, f"Expected ValueError with message '{expected_error_message}', but got {str(e)}"

    optimizer.config.exec_types[:3] = [ExecType.SEQ, ExecType.PRIM, ExecType.SEQ]
    expected_error_message = "Execution types must be in order: PAR -> SEQ -> PRIM."
    try:
        optimizer.verify()
        assert False, "Expected ValueError for invalid execution type order, but no error was raised."
    except ValueError as e:
        assert str(e) == expected_error_message, f"Expected ValueError with message '{expected_error_message}', but got {str(e)}"

    optimizer.config.exec_types[:3] = [ExecType.SEQ, ExecType.SEQ, ExecType.PAR]
    expected_error_message = "Execution types must be in order: PAR -> SEQ -> PRIM."
    try:        
        optimizer.verify()
        assert False, f"Expected ValueError for invalid execution type order, but no error was raised."
    except ValueError as e:
        assert str(e) == expected_error_message, f"Expected ValueError with message '{expected_error_message}', but got {str(e)}"
    print("test_verify passed!")

    optimizer.config.exec_types = [ExecType.SEQ] * 4 + [ExecType.PRIM] * 2
    expected_error_message = "The rightmost dimensions must be PRIM and the PRIM dimensions must include at least one dimension of each type M, N, and K."   
    try:
        optimizer.verify()
        assert False, f"Expected ValueError for missing PRIM dimension, but no error was raised."
    except ValueError as e:
        assert str(e) == expected_error_message, f"Expected ValueError with message '{expected_error_message}', but got {str(e)}"
    print("test_verify for missing PRIM dimension passed!")



if __name__ == "__main__":
    test_split_dim()
    test_fuse_dims()
    test_split_fuse_dims()
    test_permute_dims()
    test_verify()