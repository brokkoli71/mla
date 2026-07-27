MLIR_AIE_DIR ?= $(shell python3 -c "from aie.utils.config import root_path; print(root_path())")
AIE_INC      := $(MLIR_AIE_DIR)/include

PEANO_CC     = $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf -I $(AIE_INC)

.PHONY: all obj_matmul run_matmul run_matmul_opt run_matmul32 run_matmul_opt32 run_matmul64 run_matmul_opt64 run_baseline run_param run_param32 run_param64 benchmark clean

PARAM_DIR = src/baseline_parameterized
PARAM_OBJ = tensor_kernel_m1xn1xk1_bf16_bf16_bf16
PARAM_MLIR = tensor_kernel_m1xn1xk1_bf16_bf16_bf16__m2_3xn2_1xk1_8

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

build/matmul_optimiert_size32.o: src/size32/matmul_optimiert.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# Same trick as size16: derive an mlir variant linking the optimized object,
# keeping src/size32/matmul.mlir as the single source of truth.
build/matmul_optimiert_size32.mlir: src/size32/matmul.mlir | build/
	sed 's/matmul_size32\.o/matmul_optimiert_size32.o/g' $< > $@

build/zero.o: src/size16/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/zero_size32.o: src/size32/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_size64.o: src/size64/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_optimiert_size64.o: src/size64/matmul_optimiert.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# Same trick as size16/size32: derive an mlir variant linking the optimized
# object, keeping src/size64/matmul.mlir as the single source of truth.
build/matmul_optimiert_size64.mlir: src/size64/matmul.mlir | build/
	sed 's/matmul_size64\.o/matmul_optimiert_size64.o/g' $< > $@

build/zero_size64.o: src/size64/zero.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_baseline.o: src/baseline/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_baseline32.o: src/baseline32/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

build/matmul_baseline64.o: src/baseline64/matmul.s | build/
	$(PEANO_CC) --target=aie2p-none-unknown-elf -c $< -o $@

# Object name must match link_with in the parameterized .mlir.
build/$(PARAM_OBJ).o: $(PARAM_DIR)/$(PARAM_OBJ).s | build/
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

build/final_matmul_opt_size32.xclbin: build/zero_size32.o build/matmul_optimiert_size32.o build/matmul_optimiert_size32.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_opt_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_opt_size32.xclbin \
	    matmul_optimiert_size32.mlir

build/final_matmul_size64.xclbin: build/zero_size64.o build/matmul_size64.o src/size64/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_size64.xclbin \
	    ../src/size64/matmul.mlir

build/final_matmul_opt_size64.xclbin: build/zero_size64.o build/matmul_optimiert_size64.o build/matmul_optimiert_size64.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_opt_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_opt_size64.xclbin \
	    matmul_optimiert_size64.mlir

build/final_matmul_baseline_size16.xclbin: build/zero.o build/matmul_baseline.o src/baseline/matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_baseline_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_baseline_size16.xclbin \
	    ../src/baseline/matmul.mlir

build/final_tensor_kernel_param.xclbin: build/$(PARAM_OBJ).o $(PARAM_DIR)/$(PARAM_MLIR).mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_tensor_kernel_param.bin \
	    --aie-generate-xclbin --xclbin-name=final_tensor_kernel_param.xclbin \
	    ../$(PARAM_DIR)/$(PARAM_MLIR).mlir

# Parameterized-kernel instances at the benchmark's square shapes:
#   size32 -> m2=n2=2 (32x32), size64 -> m2=n2=4 (64x64), k1=8 (K=64).
# Plain instances are for correctness (run_param32/64); the *_benchmark ones
# loop the compute call 100000x per launch for the benchmark's timing.
build/final_tensor_kernel_param32.xclbin: build/$(PARAM_OBJ).o $(PARAM_DIR)/$(PARAM_OBJ)__m2_2xn2_2xk1_8.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_tensor_kernel_param32.bin \
	    --aie-generate-xclbin --xclbin-name=final_tensor_kernel_param32.xclbin \
	    ../$(PARAM_DIR)/$(PARAM_OBJ)__m2_2xn2_2xk1_8.mlir

build/final_tensor_kernel_param64.xclbin: build/$(PARAM_OBJ).o $(PARAM_DIR)/$(PARAM_OBJ)__m2_4xn2_4xk1_8.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_tensor_kernel_param64.bin \
	    --aie-generate-xclbin --xclbin-name=final_tensor_kernel_param64.xclbin \
	    ../$(PARAM_DIR)/$(PARAM_OBJ)__m2_4xn2_4xk1_8.mlir

build/final_tensor_kernel_param_benchmark_size32.xclbin: build/$(PARAM_OBJ).o $(PARAM_DIR)/$(PARAM_OBJ)__m2_2xn2_2xk1_8_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_tensor_kernel_param_benchmark_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_tensor_kernel_param_benchmark_size32.xclbin \
	    ../$(PARAM_DIR)/$(PARAM_OBJ)__m2_2xn2_2xk1_8_benchmark.mlir

build/final_tensor_kernel_param_benchmark_size64.xclbin: build/$(PARAM_OBJ).o $(PARAM_DIR)/$(PARAM_OBJ)__m2_4xn2_4xk1_8_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_tensor_kernel_param_benchmark_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_tensor_kernel_param_benchmark_size64.xclbin \
	    ../$(PARAM_DIR)/$(PARAM_OBJ)__m2_4xn2_4xk1_8_benchmark.mlir

# ── Benchmark xclbin targets ─────────────────────────────────────────────────
#
# Same compiled kernels as above (matmul.o/zero.o etc. are unchanged), just
# wired into a .mlir that repeats conv+matmul many times per host launch so
# a single kernel()+wait() amortizes host dispatch overhead over many
# on-chip iterations. Their output is not numerically correct (the
# accumulator isn't re-zeroed between reps) -- these are for timing only.

build/final_matmul_benchmark_size16.xclbin: build/zero.o build/matmul.o src/size16/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_size16.xclbin \
	    ../src/size16/matmul_benchmark.mlir

build/final_matmul_benchmark_size32.xclbin: build/zero_size32.o build/matmul_size32.o src/size32/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_size32.xclbin \
	    ../src/size32/matmul_benchmark.mlir

build/final_matmul_benchmark_size64.xclbin: build/zero_size64.o build/matmul_size64.o src/size64/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_size64.xclbin \
	    ../src/size64/matmul_benchmark.mlir

build/final_matmul_benchmark_baseline_size16.xclbin: build/zero.o build/matmul_baseline.o src/baseline/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_baseline_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_baseline_size16.xclbin \
	    ../src/baseline/matmul_benchmark.mlir

# baseline32/baseline64 are *not* correct size32/size64 matmuls -- they reuse
# the size16 baseline's 16x64/64x16/16x16 data movement but pad the matmul
# kernel's inner k-loop (.rept 2 / .rept 4) to approximate how long a real
# size32/size64 fused kernel's compute would take. Estimates only.

build/final_matmul_benchmark_baseline_size32.xclbin: build/zero.o build/matmul_baseline32.o src/baseline32/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_baseline_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_baseline_size32.xclbin \
	    ../src/baseline32/matmul_benchmark.mlir

build/final_matmul_benchmark_baseline_size64.xclbin: build/zero.o build/matmul_baseline64.o src/baseline64/matmul_benchmark.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_baseline_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_baseline_size64.xclbin \
	    ../src/baseline64/matmul_benchmark.mlir

# conv/matmul-split benchmark xclbins: same objects again, but each .mlir
# loops only conv OR only matmul %c_reps times (the other kernel runs once),
# so timing each separately shows how the total splits between the two.

build/final_matmul_benchmark_conv_size16.xclbin: build/zero.o build/matmul.o src/size16/matmul_benchmark_conv.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_conv_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_conv_size16.xclbin \
	    ../src/size16/matmul_benchmark_conv.mlir

build/final_matmul_benchmark_matmul_size16.xclbin: build/zero.o build/matmul.o src/size16/matmul_benchmark_matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_matmul_size16.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_matmul_size16.xclbin \
	    ../src/size16/matmul_benchmark_matmul.mlir

build/final_matmul_benchmark_conv_size32.xclbin: build/zero_size32.o build/matmul_size32.o src/size32/matmul_benchmark_conv.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_conv_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_conv_size32.xclbin \
	    ../src/size32/matmul_benchmark_conv.mlir

build/final_matmul_benchmark_matmul_size32.xclbin: build/zero_size32.o build/matmul_size32.o src/size32/matmul_benchmark_matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_matmul_size32.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_matmul_size32.xclbin \
	    ../src/size32/matmul_benchmark_matmul.mlir

build/final_matmul_benchmark_conv_size64.xclbin: build/zero_size64.o build/matmul_size64.o src/size64/matmul_benchmark_conv.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_conv_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_conv_size64.xclbin \
	    ../src/size64/matmul_benchmark_conv.mlir

build/final_matmul_benchmark_matmul_size64.xclbin: build/zero_size64.o build/matmul_size64.o src/size64/matmul_benchmark_matmul.mlir | build/
	cd build && aiecc.py --alloc-scheme=bank-aware --no-compile-host \
	    --no-xchesscc --no-xbridge --peano $(PEANO_INSTALL_DIR) \
	    --aie-generate-npu-insts --npu-insts-name=insts_matmul_benchmark_matmul_size64.bin \
	    --aie-generate-xclbin --xclbin-name=final_matmul_benchmark_matmul_size64.xclbin \
	    ../src/size64/matmul_benchmark_matmul.mlir

# ── Run targets ───────────────────────────────────────────────────────────────

run_matmul: build/final_matmul_size16.xclbin
	python3 src/size16/driver.py

run_matmul_opt: build/final_matmul_opt_size16.xclbin
	MATMUL_XCLBIN=build/final_matmul_opt_size16.xclbin \
	MATMUL_INSTS=build/insts_matmul_opt_size16.bin \
	python3 src/size16/driver.py

run_matmul32: build/final_matmul_size32.xclbin
	python3 src/size32/driver.py

run_matmul_opt32: build/final_matmul_opt_size32.xclbin
	MATMUL_XCLBIN=build/final_matmul_opt_size32.xclbin \
	MATMUL_INSTS=build/insts_matmul_opt_size32.bin \
	python3 src/size32/driver.py

run_matmul64: build/final_matmul_size64.xclbin
	python3 src/size64/driver.py

run_matmul_opt64: build/final_matmul_opt_size64.xclbin
	MATMUL_XCLBIN=build/final_matmul_opt_size64.xclbin \
	MATMUL_INSTS=build/insts_matmul_opt_size64.bin \
	python3 src/size64/driver.py

run_baseline: build/final_matmul_baseline_size16.xclbin
	python3 src/baseline/driver.py

run_param: build/final_tensor_kernel_param.xclbin
	python3 $(PARAM_DIR)/driver.py

run_param32: build/final_tensor_kernel_param32.xclbin
	TK_M2=2 TK_N2=2 TK_K2=8 \
	TK_XCLBIN=build/final_tensor_kernel_param32.xclbin \
	TK_INSTS=build/insts_tensor_kernel_param32.bin \
	python3 $(PARAM_DIR)/driver.py

run_param64: build/final_tensor_kernel_param64.xclbin
	TK_M2=4 TK_N2=4 TK_K2=8 \
	TK_XCLBIN=build/final_tensor_kernel_param64.xclbin \
	TK_INSTS=build/insts_tensor_kernel_param64.bin \
	python3 $(PARAM_DIR)/driver.py

benchmark: build/final_matmul_size16.xclbin build/final_matmul_size32.xclbin build/final_matmul_size64.xclbin build/final_matmul_baseline_size16.xclbin \
           build/final_matmul_benchmark_size16.xclbin build/final_matmul_benchmark_size32.xclbin build/final_matmul_benchmark_size64.xclbin build/final_matmul_benchmark_baseline_size16.xclbin \
           build/final_tensor_kernel_param_benchmark_size32.xclbin build/final_tensor_kernel_param_benchmark_size64.xclbin \
           build/final_matmul_benchmark_conv_size16.xclbin build/final_matmul_benchmark_matmul_size16.xclbin \
           build/final_matmul_benchmark_conv_size32.xclbin build/final_matmul_benchmark_matmul_size32.xclbin \
           build/final_matmul_benchmark_conv_size64.xclbin build/final_matmul_benchmark_matmul_size64.xclbin
	python3 src/benchmark/benchmark.py

# ── Utility ───────────────────────────────────────────────────────────────────

build/:
	mkdir -p build

clean:
	rm -rf build
