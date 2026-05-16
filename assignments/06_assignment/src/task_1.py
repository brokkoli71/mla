import numpy as np
import torch
import opt_einsum # unused but required for torch.einsum memory optimization
import matplotlib.pyplot as plt
from pathlib import Path

def plot_tensor(
    tensor,
    path='tensor_plot.png',
    title=''
):
    """
    Plots a 5D tensor by slicing along the first two dimensions and displaying the resulting images.
    Dimension order is assumed to be (a, b, c, y, x) where a and b are image indices and c is the color channel.

    Args:
        tensor (torch.Tensor): A 5D tensor of shape (a, b, c, y, x).
        title (str): Title for the plot.
    """
    a, b, c, y, x = tensor.shape
    fig, axes = plt.subplots(a, b, figsize=(b * 2, a * 2))
    for i in range(a):
        for j in range(b):
            img = tensor[i, j].numpy()
            # reorder from c,y,x to y,x,c
            img = np.transpose(img, (1, 2, 0))
            img *= 255.0
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

if __name__ == "__main__":
    
    file_dir = Path(__file__).parent
    
    # Download the dataset first
    #wget -O dataset.zip "https://cloud.uni-jena.de/s/4aeP53cgxoiXQEp/download"

    # Load last two intermediate tensors from disk
    print("Loading intermediate tensors from disk...")
    data = np.load(file_dir / '../data' / 'lf_tr_64_intermediate.npz')
    tensor_acspx = torch.tensor(data['tensor_acspx'])
    tensor_bspy = torch.tensor(data['tensor_bspy'])

    # Convert all tensors to torch tensors and move them to the GPU before calling `torch.einsum`. Run the contraction **twice**: once with `torch.float32` inputs and once with `torch.float16` inputs (cast the tensors before contracting).
    einsum_string = 'acspx,bspy->abcyx'
    
    #print(tensor_acspx.dtype, tensor_bspy.dtype)

    tensor_acspx_32 = tensor_acspx.to('cuda')
    tensor_bspy_32 = tensor_bspy.to('cuda')
    
    tensor_acspx_16 = tensor_acspx.to('cuda').to(torch.float16)
    tensor_bspy_16 = tensor_bspy.to('cuda').to(torch.float16)

    tensor_abcyx = torch.einsum(einsum_string, tensor_acspx_32, tensor_bspy_32)
    tensor_abcyx_16 = torch.einsum(einsum_string, tensor_acspx_16, tensor_bspy_16)

    plot_tensor(
        tensor_abcyx.to('cpu'),
        path=file_dir / 'results' / 'torch_32.png',
        title='Lightfield Tensorring Decomposition - PyTorch'
    )

    plot_tensor(
        tensor_abcyx_16.to('cpu'),
        path=file_dir / 'results' / 'torch_16.png',
        title='Lightfield Tensorring Decomposition - PyTorch (Float16)'
    )

    print( "Finished." )
