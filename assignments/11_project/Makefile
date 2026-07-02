MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
AIE_INC      := $(MLIR_AIE_DIR)/include

PEANO_CC     = $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -I $(AIE_INC)

.PHONY: all obj_matmul run_matmul16 clean

# ── Object file targets ───────────────────────────────────────────────────────

obj_matmul: build/matmul.o

build/matmul.o: src/size16/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/zero.o: src/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# ── xclbin targets ────────────────────────────────────────────────────────────


build/final_matmul_size16.xclbin: build/zero.o build/matmul.o src/size16/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=basic-sequential --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_size16.xclbin \
	    ../src/size16/matmul.mlir

# ── Run targets ───────────────────────────────────────────────────────────────

run_matmul: build/final_matmul_size16.xclbin
	python3 src/size16/driver.py

# ── Utility ───────────────────────────────────────────────────────────────────

build/:
	mkdir -p build

clean:
	rm -rf build
