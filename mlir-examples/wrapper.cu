#include <cuda.h>
#include <iostream>
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
  checkCudaErrors(cuInit(0));

  CUdevice dev;
  checkCudaErrors(cuDeviceGet(&dev, 0));

  CUcontext ctx;
  checkCudaErrors(cuCtxCreate(&ctx, 0, dev));

  // Load module from in-memory PTX string
  CUmodule module;
  checkCudaErrors(cuModuleLoadData(&module, gpu_bin_cst));
  dump_ptx_to_file(gpu_bin_cst, "kernel.ptx");
  CUfunction kernel;

  checkCudaErrors(cuModuleGetFunction(&kernel, module,
                                      "kernel_print")); // Match kernel name

  void *args[] = {};
  checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, // grid
                                 4, 1, 1,         // block
                                 0,               // shared mem
                                 0,               // stream
                                 args, 0));
  cuModuleUnload(module);
  cuCtxDestroy(ctx);

  return 0;
}
