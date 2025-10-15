## Use external crates

Currently, calling a device function from another crate is supported only through inlined functions.

We support two ways to call functions from external crates in GPU code.

### Using Real GPU Device Functions from Controllable Crates

If you control the external crate, you can make it GPU-compatible directly.

Add the gpu crate as a dependency, and mark the device functions as #[inline] to allow inlining and visibility in GPU code generation.

This is the ideal way to use device function from external crate since the implementation for CPU may not work for GPU well.

### Using CPU Functions from Uncontrolled Crates

If you want to offload CPU code to the GPU but depend on a crate you don’t control, you can still make its functions available on the GPU.
Set the GPU_EXTERNAL_CRATES environment variable to include that crate’s name.

When a crate is listed in GPU_EXTERNAL_CRATES, all its functions are treated as #[inline], making their MIR visible to the caller during GPU compilation.

However, this only works if the CPU code itself is compatible with GPU execution.

For example, translation will fail if the code:

* Uses inline CPU assembly.
* Relies on dynamic traits or function pointers.
* Depends on the global allocator (e.g., uses Box, Vec, or heap allocation).

That is, developers should only list a crate in GPU_EXTERNAL_CRATES if the functions they use are pure compute functions—that is, functions without side effects, heap allocations, or CPU-specific operations.