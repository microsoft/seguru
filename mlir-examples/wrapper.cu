#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

extern "C" const char gpu_bin_cst[];
void check(CUresult err, const char *func, const char *file, int line) {
  if (err != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(err, &msg);
    std::cerr << "CUDA error at " << file << ":" << line << " code=" << err
              << " \"" << func << "\" - " << msg << std::endl;
    exit(1);
  }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define MAX_PTX_SIZE 5000
// Dump a PTX string to a file
void dump_ptx_to_file(const char *ptx, const char *filename) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open %s for writing\n", filename);
    return;
  }

  size_t len = strlen(ptx);
  fwrite(ptx, 1, MAX_PTX_SIZE, f);
  fclose(f);
}

int main() {
  CUdeviceptr d_a, d_b; // u32[array_size]
  uint64_t array_size = 4;
  uint64_t index_to_copy = 2;
  uint64_t index_to_copy_size = 8;

  char h_a[4] = { 1, 2, 3, 4 };
  char h_b[4] = { 5, 6, 7, 8 };

  checkCudaErrors(cuInit(0));

  CUdevice dev;
  checkCudaErrors(cuDeviceGet(&dev, 0));

  CUcontext ctx;
  checkCudaErrors(cuCtxCreate(&ctx, 0, dev));

  checkCudaErrors(cuMemAlloc(&d_a, array_size * sizeof(char)));
  checkCudaErrors(cuMemAlloc(&d_b, array_size * sizeof(char)));

  checkCudaErrors(cuMemcpyHtoD(d_a, h_a, sizeof(char) * 4));
  checkCudaErrors(cuMemcpyHtoD(d_b, h_b, sizeof(char) * 4));

  // Load module from in-memory PTX string
  CUmodule module;
  checkCudaErrors(cuModuleLoadData(&module, gpu_bin_cst));
  dump_ptx_to_file(gpu_bin_cst, "kernel.ptx");
  CUfunction kernel;

  checkCudaErrors(cuModuleGetFunction(&kernel, module,
                                      "kernel_arith")); // Match kernel name

  void *args[] = {
    &d_a,
    &d_a,
    &array_size,
    &array_size,
    &array_size,
    &array_size,
    &d_b,
    &d_b,
    &array_size,
    &array_size,
    &array_size,
    &array_size
  };
  checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, // grid
                                 4, 1, 1,         //  block
                                 0,               // shared mem
                                 0,               // stream
                                 args, NULL));

  checkCudaErrors(cuMemcpyDtoH(h_b, d_b, sizeof(char) * 4));

  cudaDeviceSynchronize();

  for (int i = 0; i < 4; i++)
    printf("[?] b[%ld] = %d %c= a[%ld] = %d\n", i, h_b[i], ((h_b[i] == h_a[i]) ? '=' : '!'), i, h_a[i]);

  cuModuleUnload(module);
  cuCtxDestroy(ctx);

  return 0;
}
