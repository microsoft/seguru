NVCC ?= nvcc
CUDA_ARCH ?= sm_80
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH)

.PHONY: all clean

all: kernelbench-cuda

kernelbench-cuda: baseline.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f kernelbench-cuda
