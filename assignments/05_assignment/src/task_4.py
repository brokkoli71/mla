from config import generate_config
from pathlib import Path
import torch

def main():
    einsum = "cmk, ckn -> cmn"
    c = 4 
    m = n = k = 4096
    input_shapes = [(c, m, k), (c, k, n)]
    config = generate_config(einsum, input_shapes)
    file_dir = Path(__file__).parent
    with open(file_dir / "task_4a.out", "w") as f:
        f.write(str(config))

    L2_cache_size = torch.cuda.get_device_properties(0).L2_cache_size
    print(f"L2 cache size: {L2_cache_size / (1024**2)} MB")
    

if __name__ == "__main__":
    main()