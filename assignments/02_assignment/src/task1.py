import cupy as cp

def main():
    print("CUDA Device Attributes:" )
    for key, value in cp.cuda.Device().attributes.items():
        if key in ["L2CacheSize", "MaxSharedMemoryPerMultiprocessor", "ClockRate"]:
            print(f"\t{key}: {value}")

if __name__ == "__main__":
    main()