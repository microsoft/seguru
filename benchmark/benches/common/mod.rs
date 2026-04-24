use criterion::Criterion;
use rand::Rng;

mod empty;

use empty::Empty;

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn rand_f32_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>()).collect()
}

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn rand_float4_vec(n: usize) -> Vec<gpu::Float4> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            gpu::Float4::new([
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
            ])
        })
        .collect()
}

#[allow(dead_code)]
pub fn rand_i32_vec(n: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<i32>()).collect()
}

#[allow(dead_code)]
pub fn rand_i32_in_vocab_vec(n: usize, vocab_size: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..n).map(|_| (rng.random::<u32>() % vocab_size as u32) as i32).collect()
}

pub trait KernelRunner<'a>: Sized {
    fn new<N: gpu_host::GpuCtxSpace>(
        ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self>;

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig;

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    );

    #[allow(dead_code)]
    fn c_fn(&mut self);
}

#[allow(dead_code)]
pub struct Config {
    pub batch_size: usize,
    pub seq_len: usize,
    pub channel: usize,
    pub out_channel: usize,
    pub vocab_size: usize,
    pub padded_vocab_size: usize,
    pub head_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

impl Config {
    fn to_str(&self) -> String {
        format!(
            "{}_{}_{}_{}_{}_{}_scvpho",
            self.seq_len,
            self.channel,
            self.vocab_size,
            self.padded_vocab_size,
            self.head_size,
            self.out_channel
        )
    }

    #[allow(dead_code)]
    pub fn get_params_sizes(&self) -> [usize; 16] {
        let ch = self.channel;
        let num_layers = self.num_layers;
        [
            self.padded_vocab_size * ch, // wte
            self.seq_len * ch,           // wpe
            num_layers * ch,             // ln1w
            num_layers * ch,             // ln1b
            num_layers * (3 * ch) * ch,  // qkvw
            num_layers * (3 * ch),       // qkvb
            num_layers * ch * ch,        // attprojw
            num_layers * ch,             // attprojb
            num_layers * ch,             // ln2w
            num_layers * ch,             // ln2b
            num_layers * (4 * ch) * ch,  // fcw
            num_layers * (4 * ch),       // fcb
            num_layers * ch * (4 * ch),  // fcprojw
            num_layers * ch,             // fcprojb
            ch,                          // lnfw
            ch,                          // lnfb
        ]
    }
}

pub fn bench_llm_rs<'a, N: gpu_host::GpuCtxSpace, B: KernelRunner<'a>>(
    c: &mut Criterion,
    name: &str,
    ctx: &'a gpu_host::GpuCtxGuard<N>,
    m: &'a gpu_host::GpuModule<N>,
) {
    let mut group = c.benchmark_group(name);
    let batch_size = 1;
    for seq_length_order in [10, 14, 20] {
        let seq_len = 1 << seq_length_order;
        for out_channel in [128] {
            for vocab_size in [1024] {
                let num_heads = 8;
                let channel = out_channel / 4;
                let head_size = channel / num_heads;
                let config = Config {
                    batch_size,
                    seq_len,
                    vocab_size,
                    padded_vocab_size: vocab_size,
                    head_size,
                    channel,
                    out_channel,
                    num_heads,
                    num_layers: 12,
                };
                let config_str = config.to_str();
                let Some(mut mybench) = B::new(ctx, config) else {
                    continue;
                };
                use gpu_host::SafeGpuConfig;
                let launch_config = mybench.launch_config();
                let gdim_x = launch_config.grid_dim_x();
                let bdim_x = launch_config.block_dim_x();
                let gdim_y = launch_config.grid_dim_y();
                let bdim_y = launch_config.block_dim_y();
                let smem = launch_config.shared_size();
                let config_str =
                    format!("{}_{}_{}_{}_{}_{}", config_str, gdim_x, bdim_x, gdim_y, bdim_y, smem);

                // Run empty benchmark to get launch overhead
                let mut empty_bench = Empty::new_from_runner(&mybench).unwrap();
                // if feature seguru is enabled
                #[cfg(feature = "seguru")]
                group.bench_function(format!("emptyrs_{}", config_str).as_str(), |b| {
                    b.iter(|| {
                        empty_bench.rs_fn(ctx, m);
                        let _ = ctx.sync();
                    })
                });
                // if nvvm or llvm feature is enabled
                #[cfg(any(feature = "nvvm", feature = "llvm"))]
                let c_prefix = if cfg!(feature = "nvvm") {
                    "nvvm"
                } else if cfg!(feature = "llvm") {
                    "llvm"
                } else {
                    "c"
                };
                #[cfg(any(feature = "nvvm", feature = "llvm"))]
                group.bench_function(format!("empty{}_{}", c_prefix, config_str).as_str(), |b| {
                    b.iter(|| {
                        empty_bench.c_fn();
                        let _ = ctx.sync();
                    })
                });

                // Run actual benchmark
                // add wbound or wobound depending on DISABLE_GPU_BOUND_CHECK env var.
                #[cfg(feature = "seguru")]
                let disable_gpu_bound_check = std::env::var("DISABLE_GPU_BOUND_CHECK")
                    .map(|value| value == "true")
                    .unwrap_or(false);
                #[cfg(feature = "seguru")]
                group.bench_function(
                    format!(
                        "{}_{}",
                        if disable_gpu_bound_check { "rsnobound" } else { "rs" },
                        config_str
                    )
                    .as_str(),
                    |b| {
                        b.iter(|| {
                            mybench.rs_fn(ctx, m);
                            let _ = ctx.sync();
                        })
                    },
                );
                #[cfg(any(feature = "nvvm", feature = "llvm"))]
                group.bench_function(format!("{}_{}", c_prefix, config_str).as_str(), |b| {
                    b.iter(|| {
                        mybench.c_fn();
                        let _ = ctx.sync();
                    })
                });
            }
        }
    }
    group.finish();
}

#[macro_export]
macro_rules! gen_bench {
    ($($bench_type:ty, $name:expr),*) => {
        fn bench_function(c: &mut Criterion) {
            gpu_host::cuda_ctx(0, |ctx, m| {
                $(bench_llm_rs::<_, $bench_type>(c, $name, ctx, m);)*
            });
        }

        criterion_group! {
          name = bench;
          config = Criterion::default().warm_up_time(Duration::from_secs(2)).measurement_time(Duration::from_secs(3)).without_plots();
          targets = bench_function
        }

        criterion_main!(bench);
    };
}
