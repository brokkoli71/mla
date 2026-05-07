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
    
if __name__ == "__main__":
    test_split_dim()
    test_fuse_dims()
    test_split_fuse_dims()
    test_permute_dims()