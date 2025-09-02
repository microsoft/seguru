use gpu_host::cuda_ctx;
use syntax_host::run_host_arith;

#[test]
#[should_panic(expected = "Failed to get device")]
fn test_fails() {
    cuda_ctx(111, |ctx| {
        run_host_arith(ctx, 4, 1).expect("Failed to run host arithmetic");
    });
}
