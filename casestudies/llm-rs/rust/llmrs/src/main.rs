#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(clippy::too_many_arguments)]

use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize};

use clap::Parser;
use cudarc::cublas::sys as cublas_sys;
use gpu_host::{GpuCtxGuard, GpuCtxSpace, GpuModule, cuda_ctx};

#[macro_use]
mod model;
mod tokenizer;

use model::GPT2;
use model::dataloader::DataLoader;

macro_rules! top_path {
    ($p: literal) => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../", $p)
    };
}

#[macro_export]
macro_rules! time_it {
    ($func:expr) => {{
        let start = std::time::Instant::now();
        let result = $func;
        println!("{:?}: {}", stringify!($func), start.elapsed());
        result
    }};
}

#[derive(Parser, Debug)]
#[command(name = "llmrs")]
struct Args {
    /// Name of the user
    #[arg(default_value = top_path!("dev/data/tinyshakespeare/tiny_shakespeare_train.bin"), long)]
    train_data_pattern: PathBuf,
    #[arg(default_value = top_path!("dev/data/tinyshakespeare/tiny_shakespeare_val.bin"), long)]
    val_data_pattern: PathBuf,
    #[arg(default_value = top_path!("gpt2_124M.bin"), value_parser, long)]
    model_path: PathBuf,
    #[arg(default_value = top_path!("gpt2_tokenizer.bin"), value_parser, long)]
    tokenizer_path: PathBuf,
    #[arg(default_value = "llm_rs.log", long)]
    output_log_file: String,
    #[arg(default_value_t = 4, long)]
    batch_size: usize,
    #[arg(default_value_t = 1024, long)]
    seq_length: usize,
    #[arg(default_value_t = 3e-4, long)]
    learning_rate: f32,
    #[arg(default_value_t = 20, long)]
    val_loss_every: usize,
    #[arg(default_value_t = 20, long)]
    val_max_steps: usize,
    #[arg(default_value_t = 20, long)]
    sample_every: usize,
    #[arg(default_value_t = 64, long)]
    gen_t: usize,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    println!("params: {args:?}");
    cuda_ctx(0, |ctx, m| {
        llm_rs_run(ctx, m, &args);
    });
}

#[derive(Debug)]
pub struct UnsafeCudaContext {
    pub(crate) cu_device: cudarc::driver::sys::CUdevice,
    pub(crate) cu_ctx: *mut std::ffi::c_void,
    pub(crate) ordinal: usize,
    pub(crate) has_async_alloc: bool,
    pub(crate) num_streams: AtomicUsize,
    pub(crate) event_tracking: AtomicBool,
    pub(crate) error_state: AtomicU32,
}

const GPT2_EOT: i32 = 50256;

pub fn sample_softmax(logits: &[f32], coin: f32) -> i32 {
    // coin should be in [0, 1)
    assert!(!logits.is_empty());
    assert!((0.0..1.0).contains(&coin));

    // compute normalization factor
    let norm: f64 = logits.iter().map(|&x| (x as f64).exp()).sum();

    // scale the coin
    let coin = coin * norm as f32;

    let mut cdf = 0.0f32;
    for (i, &x) in logits.iter().enumerate() {
        cdf += x.exp();
        if coin < cdf {
            return i as i32;
        }
    }
    logits.len() as i32 - 1 // fallback in case of rounding
}

pub fn safe_print(piece: &str) {
    // handle single-byte tokens specially
    if piece.len() == 1 {
        let b = piece.as_bytes()[0];
        if !b.is_ascii_graphic() && !b.is_ascii_whitespace() {
            print!("{piece}");
            return; // skip non-printable
        }
    }

    print!("{piece}");
}

fn llm_rs_run<'ctx, 'a, NS: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, 'a, NS>,
    m: &GpuModule<NS>,
    args: &Args,
) {
    let rng = rand::rng();
    let cublas_handle = {
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cublas_sys::cublasCreate_v2(handle.as_mut_ptr());
            handle.assume_init()
        }
    };
    let (major, minor) = ctx.get_compute_capability();
    let enable_tf32 = major >= 8;
    unsafe {
        cublas_sys::cublasSetMathMode(
            cublas_handle,
            if enable_tf32 {
                cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH
            } else {
                cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH
            },
        );
    }
    println!("cublas handle: {:?}, TF32: {}", cublas_handle, enable_tf32);
    let mut model = GPT2::new(ctx, m, &args.model_path).unwrap_or_else(|_| {
        panic!("Error initializing model from checkpoint");
    });
    println!("| max_sequence_length T | {} |\n", model.config.max_seq_len);
    println!("| vocab_size V          | {} |\n", model.config.vocab_size);
    println!("| padded_vocab_size Vp  | {} |\n", model.config.padded_vocab_size);
    println!("| num_layers L          | {} |\n", model.config.num_layers);
    println!("| num_heads NH          | {} |\n", model.config.num_heads);
    println!("| channels C            | {} |\n", model.config.channels);
    println!("| num_parameters        | {} |\n", model.num_parameters);
    println!("+-----------------------+----------------------------------------------------+\n");
    let mut train_loader =
        DataLoader::new(&args.train_data_pattern, args.batch_size, args.seq_length);
    let mut val_loader = DataLoader::new(&args.val_data_pattern, args.batch_size, args.seq_length);
    /*
    int val_num_batches = val_loader.num_tokens / (B*T);
    if (val_num_batches > val_max_steps) { val_num_batches = val_max_steps; }
    */
    let val_num_batches = val_loader.num_tokens / (args.batch_size * args.seq_length);
    let val_num_batches =
        if val_num_batches > args.val_max_steps { args.val_max_steps } else { val_num_batches };
    println!("| train_num_batches     | {} |\n", train_loader.num_batches);
    println!("val_num_batches: {}", val_num_batches);
    println!("+-----------------------+----------------------------------------------------+\n");
    println!(
        "allocated {} MiB for model parameters",
        (model.num_parameters * std::mem::size_of::<f32>()) / (1024 * 1024)
    );

    let mut tokenizer = tokenizer::Tokenizer::new(&args.tokenizer_path);
    let padded_vocab_size = model.config.padded_vocab_size;
    let vocab_size = model.config.vocab_size;
    // some memory for generating samples from the model
    let rng_state: u64 = 1337;
    let mut gen_tokens = vec![0i32; args.batch_size * args.seq_length];
    let mut cpu_logits = vec![0.0f32; vocab_size];

    // train
    for step in 0..=train_loader.num_batches {
        let last_step = step == train_loader.num_batches;

        // once in a while estimate the validation loss
        if step % args.val_loss_every == 0 || last_step {
            let mut val_loss = 0.0f32;
            val_loader.reset();
            for _ in 0..val_num_batches {
                let (input, target) = val_loader.next_batch();
                model.forward(cublas_handle, &input, &target, args.batch_size, args.seq_length);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches as f32;
            println!("val loss {val_loss}");
            //logger_log_val(&logger, step, val_loss);
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % args.sample_every == 0) || last_step {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            gen_tokens.iter_mut().for_each(|t| *t = GPT2_EOT); // GPT2_EOT
            // now sample from the model autoregressively
            println!("generating:\n---");
            for t in 1..args.gen_t {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                model.forward(cublas_handle, &gen_tokens, &[], args.batch_size, args.seq_length);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // only using position 0 because it's a bit faster (copy less probs from GPU -> CPU)
                // get the V-dimensional vector probs[0, t-1, :]
                let mut acts = model.acts.as_mut().unwrap().inner();
                let output_offset = (t - 1) * padded_vocab_size;
                let logits = acts.output.index_mut(output_offset..(output_offset + vocab_size)); // first row
                logits.copy_to_host(&mut cpu_logits).unwrap();
                // float coin = random_f32(&rng_state);
                let coin = 0.5; //rng.gen_range(0.0..1.0);
                let next_token = sample_softmax(&cpu_logits, coin);
                gen_tokens[t] = next_token;
                //println!("next token: {}\n", next_token);
                if tokenizer.init_ok {
                    let token_str = tokenizer.decode(next_token as u32);
                    safe_print(token_str);
                } else {
                    // fall back to printing the token id
                    println!("{} ", next_token);
                }
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }
            println!("\n---\n");
        }

        if last_step {
            break;
        }

        let start = std::time::Instant::now();
        let (input, target) = train_loader.next_batch();
        model.forward(cublas_handle, &input, &target, args.batch_size, args.seq_length);
        model.zero_grad();
        model.backward(cublas_handle);
        model.update(args.learning_rate, 0.9, 0.999, 1e-8, 0.0, (step + 1) as i32);
        let _ = ctx.sync();
        let elapsed = start.elapsed();
        let tokens_per_second = (args.batch_size * args.seq_length) as f32 / elapsed.as_secs_f32();
        println!(
            "step {}/{}: train loss {} ({:?}, {} tok/s)",
            step + 1,
            train_loader.num_batches,
            model.mean_loss,
            elapsed,
            tokens_per_second as usize
        );
        //logger_log_train(&logger, step, model.mean_loss);
    }
}
