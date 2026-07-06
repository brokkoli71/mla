MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
AIE_INC      := $(MLIR_AIE_DIR)/include

PEANO_CC     = $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -I $(AIE_INC)

.PHONY: all obj_matmul run_matmul run_matmul_opt run_matmul32 run_matmul64 run_baseline benchmark clean

# ── Object file targets ───────────────────────────────────────────────────────

obj_matmul: build/matmul.o

build/matmul.o: src/size16/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_optimiert.o: src/size16/matmul_optimiert.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# Derive an mlir variant that links the optimized object instead of matmul.o,
# keeping src/size16/matmul.mlir as the single source of truth.
build/matmul_optimiert.mlir: src/size16/matmul.mlir | build/
	sed 's/matmul\.o/matmul_optimiert.o/g' $< > $@

build/matmul_size32.o: src/size32/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/zero.o: src/size16/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/zero_size32.o: src/size32/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_size64.o: src/size64/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/zero_size64.o: src/size64/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_baseline.o: src/baseline/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# ── xclbin targets ────────────────────────────────────────────────────────────

build/final_matmul_size16.xclbin: build/zero.o build/matmul.o src/size16/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_size16.xclbin \
	    ../src/size16/matmul.mlir

build/final_matmul_opt_size16.xclbin: build/zero.o build/matmul_optimiert.o build/matmul_optimiert.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_opt_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_opt_size16.xclbin \
	    matmul_optimiert.mlir

build/final_matmul_size32.xclbin: build/zero_size32.o build/matmul_size32.o src/size32/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_size32.xclbin \
	    ../src/size32/matmul.mlir

build/final_matmul_size64.xclbin: build/zero_size64.o build/matmul_size64.o src/size64/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_size64.xclbin \
	    ../src/size64/matmul.mlir

build/final_matmul_baseline_size16.xclbin: build/zero.o build/matmul_baseline.o src/baseline/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_baseline_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_baseline_size16.xclbin \
	    ../src/baseline/matmul.mlir

# ── Run targets ───────────────────────────────────────────────────────────────

run_matmul: build/final_matmul_size16.xclbin
	python3 src/size16/driver.py

run_matmul_opt: build/final_matmul_opt_size16.xclbin
	MATMUL_XCLBIN=build/final_matmul_opt_size16.xclbin \
	MATMUL_INSTS=build/insts_matmul_opt_size16.bin \
	python3 src/size16/driver.py

run_matmul32: build/final_matmul_size32.xclbin
	python3 src/size32/driver.py

run_matmul64: build/final_matmul_size64.xclbin
	python3 src/size64/driver.py

run_baseline: build/final_matmul_baseline_size16.xclbin
	python3 src/baseline/driver.py

benchmark: build/final_matmul_size16.xclbin build/final_matmul_size32.xclbin build/final_matmul_size64.xclbin build/final_matmul_baseline_size16.xclbin
	python3 src/benchmark/benchmark.py

# ── Utility ───────────────────────────────────────────────────────────────────

build/:
	mkdir -p build

clean:
	rm -rf build
