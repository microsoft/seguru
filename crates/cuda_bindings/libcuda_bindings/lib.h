#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int gpu_init(void);
void *gpu_memalloc(size_t size);
int gpu_free(void *ptr);
int gpu_device_sync(void);
int gpu_load_module(void);
int gpu_unload_module(void);
int gpu_launch_kernel(const char *func_name,
		      unsigned int gridDimX, unsigned int gridDimY,
		      unsigned int gridDimZ, unsigned int blockDimX,
		      unsigned int blockDimY, unsigned int blockDimZ,
		      unsigned int sharedMemBytes,
		      void **kernelParams, void **extra);

#ifdef __cplusplus
}
#endif