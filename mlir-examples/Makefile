MLIR_OPT = mlir-opt
MLIR_TRANSLATE = mlir-translate
LLC = llc

%.gpu.mlir: %.mlir
	@echo "Generating LLVM IR from MLIR..."
	$(MLIR_OPT) -gpu-lower-to-nvvm-pipeline='opt-level=3 cubin-chip=sm_80 cubin-features=+ptx80' $< -o $@

%.ll: %.gpu.mlir
	$(MLIR_TRANSLATE) -mlir-to-llvmir $< -o $@

%.as: %.ll
	llvm-as $< -o $@

%.o: %.as
	$(LLC) -filetype=obj $(LLC_FLAGS) $< -o $@
	objcopy --globalize-symbol=gpu_bin_cst  $@ $@

hello: wrapper.cu hello_mlir.o
	@echo "Generating wrapper..."
	nvcc -o $@  hello_mlir.o $< -lcuda -lcudart

gpu_basic: wrapper.cu ../crates/target/debug/deps/gpu_basic-*.rcgu.o
	@echo "Generating wrapper..."
	nvcc -o $@  /home/ziqiaozhou/rust-gpu/rust-gpu/crates/target/debug/deps/gpu_basic-*.rcgu.o $< -lcuda -lcudart