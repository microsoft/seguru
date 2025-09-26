#[test]
#[cfg_attr(
    not(debug_assertions),
    should_panic(expected = "Kernel execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS")
)]
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "Kernel execution failed: CUDA Error: CUDA_ERROR_ASSERT")
)]
fn test_fails_out_of_bound1() {
    syntax_host::test_oob1()
}
