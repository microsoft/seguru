use gpu_host::cuda_ctx;
use syntax_host::run_host_arith;

#[test]
#[cfg_attr(
    not(debug_assertions),
    should_panic(expected = "GPU execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS")
)]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "GPU execution failed: CUDA Error: CUDA_ERROR_ASSERT")
)]
fn test_fails_out_of_bound() {
    cuda_ctx(0, |ctx, m| run_host_arith(ctx, m, 3, 1)).expect("failed");
}
