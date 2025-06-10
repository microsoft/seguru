MLIR_OPT = mlir-opt
MLIR_TRANSLATE = mlir-translate
LLC = llc

.PHONY: clean

default: gpu_basic

%.gpu.mlir: %.mlir
	@echo "  OPT     [i]" $@
	@$(MLIR_OPT) -gpu-lower-to-nvvm-pipeline='opt-level=3 cubin-chip=sm_80 cubin-features=+ptx80' $< -o $@

%.ll: %.gpu.mlir
	@echo "  TRANS   [i]" $@
	@$(MLIR_TRANSLATE) -mlir-to-llvmir $< -o $@

%.as: %.ll
	@echo "  AS      [i]" $@
	@llvm-as $< -o $@

%.o: %.as
	@echo "  LLC    " $@
	@$(LLC) -filetype=obj $(LLC_FLAGS) $< -o $@
	@echo "  OBJCOPY" $@
	@objcopy --globalize-symbol=gpu_bin_cst  $@ $@

wrapper.o: wrapper.cu
	@echo "  NVCC   " $@
	@nvcc -c $< -o $@ -Wno-deprecated-gpu-targets

hello: wrapper.o hello_mlir.o
	@echo "  LINK   " $@
	@nvcc -o $@ $^ -lcuda -lcudart -Wno-deprecated-gpu-targets

gpu_basic: wrapper.o ../crates/target/release/deps/gpu_test_arith-*.rcgu.o
	@echo "  LINK   " $@
	@nvcc -o $@ $^ -lcuda -lcudart -Wno-deprecated-gpu-targets

clean:
	@echo "  CLEAN"
	@rm -rf *.ptx *.o gpu_basic hello