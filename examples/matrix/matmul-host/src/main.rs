use std::env;
use std::str::FromStr;

use cuda_bindings::cuda_ctx;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n: usize = 4;

    if args.len() >= 2 {
        // Take length here
        n = i32::from_str(&args[1]).unwrap() as usize;
        println!("{}: n set to {}", args[0], n);
    }
    cuda_ctx(0, |ctx| {
        matmul_host::run_host_matmul(ctx, n).expect("Failed to run host arithmetic");
    });
}
