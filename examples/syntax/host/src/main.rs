use gpu_host::cuda_ctx;
use std::env;
use std::str::FromStr;
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut len: usize = 4;
    let w: usize = 1;

    if args.len() >= 2 {
        // Take length here
        len = i32::from_str(&args[1]).unwrap() as usize;
        println!("{}: length set to {}", args[0], len);
    }
    cuda_ctx(0, |ctx| {
        syntax_host::run_host_arith(ctx, len, w).expect("Failed to run host arithmetic");
    });
}
