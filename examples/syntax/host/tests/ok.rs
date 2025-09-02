use gpu_host::cuda_ctx;
use syntax_host::run_host_arith;

#[test]
fn test_ok() {
    cuda_ctx(0, |ctx| {
        run_host_arith(ctx, 4, 1).expect("Failed to run host arithmetic");
    });
}

#[test]
fn test_ok_with_len() {
    cuda_ctx(0, |ctx| {
        run_host_arith(ctx, 10, 1).expect("Failed to run host arithmetic");
    });
}
