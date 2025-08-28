use gpu_host::cuda_ctx;
use syntax_host::run_host_arith;

#[test]
#[should_panic(expected = "Kernel execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS")]
fn test_fails_out_of_bound() {
    cuda_ctx(0, |ctx| run_host_arith(ctx, 3, 1));
}
