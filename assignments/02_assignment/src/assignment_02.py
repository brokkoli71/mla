import torch
import cuda.tile as ct
import cupy as cp



def main():
    cp.cuda.Device().attributes.items()

if __name__ == "__main__":
    main()
