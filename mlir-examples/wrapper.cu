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

#define MAX_PTX_SIZE 50000
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

int main(int argc, const char **argv) {
  CUdeviceptr d_a, d_b, d_c; // u32[array_size]
  uint64_t array_size = 4;
  uint64_t fake_array_size = array_size;
  uint64_t window = 1;

  if (argc > 2) {
    printf("usage: %s <fake array size>", argv[0]);
    return -1;
  }
  

  if (argc == 2) {
    fake_array_size = atoi(argv[1]);
  }

  int h_a[4] = { 1, 2, 3, 4 };
  int h_b[4] = { 5, 6, 7, 8 };
  int h_c[4] = { 10, 11, 12, 13};

  checkCudaErrors(cuInit(0));

  CUdevice dev;
  checkCudaErrors(cuDeviceGet(&dev, 0));

  CUcontext ctx;
  checkCudaErrors(cuCtxCreate(&ctx, 0, dev));

  checkCudaErrors(cuMemAlloc(&d_a, array_size * sizeof(int)));
  checkCudaErrors(cuMemAlloc(&d_b, array_size * sizeof(int)));
  checkCudaErrors(cuMemAlloc(&d_c, array_size * sizeof(int)));

  checkCudaErrors(cuMemcpyHtoD(d_a, h_a, sizeof(int) * 4));
  checkCudaErrors(cuMemcpyHtoD(d_b, h_b, sizeof(int) * 4));
  checkCudaErrors(cuMemcpyHtoD(d_c, h_c, sizeof(int) * 4));

  // Load module from in-memory PTX string
  CUmodule module;
  checkCudaErrors(cuModuleLoadData(&module, gpu_bin_cst));
  dump_ptx_to_file(gpu_bin_cst, "kernel.ptx");
  CUfunction kernel;
  CUfunction enumfunc[10];

  memset(enumfunc, 0, sizeof(enumfunc));

  printf("getting function names\n");
  checkCudaErrors(cuModuleEnumerateFunctions(enumfunc, 10, module));
  printf("printing function names\n");
  for (int i = 0; i < 10; i++) {
    const char *name;
    if (enumfunc[i] == NULL) {
      break;
    }
    checkCudaErrors(cuFuncGetName(&name, enumfunc[i]));
    printf("func enum[%i]: %s\n", i, name);
  }

  checkCudaErrors(cuModuleGetFunction(&kernel, module,
                                      "kernel_arith")); // Match kernel name

  void *args[] = {
    &d_a,
    &fake_array_size,
    &window,
    &d_b,
    &fake_array_size,
    &window,
    &d_c,
    &fake_array_size,
    NULL
  };

  checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, // grid
                                 4, 1, 1,         // block
                                 0,               // shared mem
                                 0,               // stream
                                 args, NULL));

  checkCudaErrors((CUresult)cudaDeviceSynchronize());

  checkCudaErrors(cuMemcpyDtoH(h_b, d_b, sizeof(int) * 4));

  cudaDeviceSynchronize();

  for (int i = 0; i < 4; i++)
    printf("[?] b[%d] = %d %c= a[%d] = %d\n", i, h_b[i], ((h_b[i] == h_a[i]) ? '=' : '!'), i, h_a[i]);

  cuModuleUnload(module);
  cuCtxDestroy(ctx);

  return 0;
}
