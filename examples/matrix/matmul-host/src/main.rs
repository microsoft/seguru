use std::env;
use std::str::FromStr;

use gpu_host::cuda_ctx;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n: usize = 32;
    let mut dim: u32 = 16;

    if args.len() >= 2 {
        // Take length here
        n = i32::from_str(&args[1]).unwrap() as usize;
        if args.len() < 3 {
            println!("{}: n set to {}", args[0], n);
        } else {
            dim = u32::from_str(&args[2]).unwrap();
            println!("{}: n set to {}, dim set to {}", args[0], n, dim);
        }
    }
    cuda_ctx(0, |ctx, m| {
        matmul_host::run_host_matmul(ctx, m, n, dim).expect("Failed to run host arithmetic");
    });
}

#[test]
fn test_naive_matmul() {
    cuda_ctx(0, |ctx, m| {
        matmul_host::run_host_matmul(ctx, m, 32, 16).expect("Failed to run host arithmetic");
    });
}
