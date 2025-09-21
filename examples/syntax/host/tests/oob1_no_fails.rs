// After enable integer overflow checks, this test will fail due to extra br,
// and the ptx compiler seems not be able to optimize code out.
// with Kernel execution failed: CUDA Error: CUDA_ERROR_ILLEGAL_ADDRESS
#[test]
#[cfg(not(debug_assertions))]
fn test_oob_no_fails() {
    syntax_host::test_oob_no_fails()
}
