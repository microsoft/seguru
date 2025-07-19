use cuda_bindings::cuda_ctx;
use manual_test_host_arith::run_host_arith;

#[test]
#[should_panic(expected = "Kernel execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS")]
fn test_fails_out_of_bound() {
    cuda_ctx(0, |ctx| run_host_arith(ctx, 3, 1));
}
