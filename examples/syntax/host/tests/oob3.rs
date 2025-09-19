#[test]
#[should_panic(expected = "Kernel execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS")]
fn test_fails_out_of_bound3() {
    syntax_host::test_oob3()
}
