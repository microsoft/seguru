#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

extern "C" const char gpu_bin_cst[];

static inline CUresult check(CUresult err, const char *func, 
			 const char *file, int line)
{
	if (err != CUDA_SUCCESS) {
		const char *msg;
		cuGetErrorName(err, &msg);
		fprintf(stderr, 
			"CUDA Driver API error at %s : %d code = %d \"%s\" - %s\n",
			file, line, err, func, msg);
	}
	return err;
}
#define checkCuErrors(val) check((val), #val, __FILE__, __LINE__)

#if 0

#define MAX_PTX_SIZE 50000
/* Dump a PTX string to a file */
static void dump_ptx_to_file(const char *ptx, const char *filename)
{
	FILE *f = fopen(filename, "wb");
	if (!f) {
		fprintf(stderr, "can't open %s for PTX dumping\n", filename);
		return;
	}

	fwrite(ptx, 1, MAX_PTX_SIZE, f);
	fclose(f);
}

#endif

static int init_done = 0;
static CUdevice dev;
static CUcontext ctx;

extern "C" int gpu_init(void)
{
	CUresult err;

	if (init_done) {
		return 0;
	}

	err = checkCuErrors(cuInit(0));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	err = checkCuErrors(cuDeviceGet(&dev, 0));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}
	
	err = checkCuErrors(cuCtxCreate(&ctx, 0, dev));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

extern "C" int gpu_destroy(void)
{
	CUresult err;

	if (!init_done) {
		return -1;
	}

	err = checkCuErrors(cuCtxDestroy(ctx));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

extern "C" unsigned long long gpu_memalloc(size_t size)
{
	CUresult err;
	unsigned long long ret = 0;

	err = checkCuErrors(cuMemAlloc(&ret, size));
	if (err != CUDA_SUCCESS) {
		return 0;
	}

	return ret;
}

extern "C" int gpu_free(unsigned long long ptr)
{
	CUresult err;

	err = checkCuErrors(cuMemFree(ptr));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

extern "C" int gpu_device_sync(void)
{
	CUresult err;

	/* TODO: not a good pratice to mix driver API and user space API! */
	err = checkCuErrors((CUresult)cudaDeviceSynchronize());
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

/* only *one* module is currently supported */
static CUmodule module;

/* 
 * TODO: DEBATE: do we still need that gpu_bin_cst or do we want the user to
 *               pass it here?
 */
extern "C" int gpu_load_module(/* const char *ptx_buffer */void)
{
	CUresult err;

	err = checkCuErrors(cuModuleLoadData(&module, gpu_bin_cst));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

extern "C" int gpu_unload_module(void)
{
	CUresult err;

	err = checkCuErrors(cuModuleUnload(module));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}

extern "C" int gpu_launch_kernel(const char *func_name,
				 unsigned int gridDimX, unsigned int gridDimY,
				 unsigned int gridDimZ, unsigned int blockDimX,
				 unsigned int blockDimY, unsigned int blockDimZ,
				 unsigned int sharedMemBytes,
				 void **kernelParams, void **extra)
{
	CUresult err;
	CUfunction kernel;

	err = checkCuErrors(cuModuleGetFunction(&kernel, module, func_name));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	err = checkCuErrors(cuLaunchKernel(kernel, gridDimX, gridDimY, gridDimZ,
					   blockDimX, blockDimY, blockDimZ,
					   sharedMemBytes,
					   0, /* CUstream */
					   kernelParams, extra));
	if (err != CUDA_SUCCESS) {
		return (int)err;
	}

	return 0;
}
