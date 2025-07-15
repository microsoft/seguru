MLIR_OPT = mlir-opt
MLIR_GPU_COMPILE = "./../crates/target/debug/mlir-gpu-compile"
MLIR_TRANSLATE = mlir-translate
LLC = llc

.PHONY: clean

default: gpu_basic

%.o: %.mlir
	@echo "  OPT     [i]" $@
	@$(MLIR_GPU_COMPILE) $<

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
	@rm -rf *.ptx *.o gpu_basic hello *.bc *.ll *.mir