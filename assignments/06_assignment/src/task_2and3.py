import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt
from pathlib import Path
import sys

file_dir = Path(__file__).parent
assignment_05_src = (file_dir / '../../05_assignment/src').resolve()
# Add it to Python's search path
sys.path.append(str(assignment_05_src))

from optimizer import Optimizer
from config import Config, DataType, PrimType, DimType, ExecType, generate_config





if __name__ == "__main__":

    # Use the `generate_config` function you implemented in Assignment 05.

    # a) Call `generate_config` with the einsum string from Task 1 and the shapes of `tensor_acspx` and `tensor_bspy` to produce an initial `Config`. You may choose either fp32 or fp16 as the data types for the config.

    # b) **Report** the resulting config (all fields).

    file_dir = Path(__file__).parent
    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    data = np.load(file_dir / '../data' / 'lf_tr_64_intermediate.npz')
    tensor_acspx = torch.tensor(data['tensor_acspx'])
    tensor_bspy = torch.tensor(data['tensor_bspy'])

    # Convert all tensors to torch tensors and move them to the GPU before calling `torch.einsum`. Run the contraction **twice**: once with `torch.float32` inputs and once with `torch.float16` inputs (cast the tensors before contracting).
    einsum_string = 'acspx,bspy->abcyx'

    tensor_acspx_32 = tensor_acspx.to('cuda')
    tensor_bspy_32 = tensor_bspy.to('cuda')
    
    tensor_acspx_16 = tensor_acspx.to('cuda').to(torch.float16)
    tensor_bspy_16 = tensor_bspy.to('cuda').to(torch.float16)

    config = generate_config(einsum_string, [tensor_acspx_16.shape, tensor_bspy_16.shape], dim_order=None)
    file_dir = Path(__file__).parent

    print(config);
    # with open(file_dir / "results" / "task2_config.out", "w") as f:
    #     f.write(str(config))

    optimized_config = Optimizer.optimize(config)


