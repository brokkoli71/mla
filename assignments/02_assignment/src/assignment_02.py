import torch
import cuda.tile as ct
import cupy as cp



def main():
    cp.cuda.Device().attributes.items()
    print("CUDA Device Attributes:" )
    for key, value in cp.cuda.Device().attributes.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
