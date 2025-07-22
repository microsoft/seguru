use std::env;
use std::str::FromStr;

use cuda_bindings::cuda_ctx;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut n: usize = 4;
    let mut dim: u32 = 2;

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
    cuda_ctx(0, |ctx| {
        matmul_host::run_host_matmul(ctx, n, dim).expect("Failed to run host arithmetic");
    });
}
