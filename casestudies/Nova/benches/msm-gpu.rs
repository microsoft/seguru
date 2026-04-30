use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use halo2curves::{group::Group, CurveAffine};
use nova_snark::provider;
use nova_snark::provider::bn256_grumpkin::{bn256, grumpkin};
use nova_snark::provider::pasta::{pallas, vesta};
use nova_snark::provider::secp_secq::{secp256k1, secq256k1};
#[cfg(feature = "blitzar")]
use provider::blitzar;
use provider::msm;
#[cfg(feature = "gpu")]
use provider::msm_gpu;
use rand_core::OsRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Duration;

// cargo criterion --bench --no-default-features --features rs_gpu --bench msm-gpu
// ---------------------------------------------------------
// Criterion setup
// ---------------------------------------------------------
criterion_group! {
  name = msm_gpu_group;
  config = Criterion::default().warm_up_time(Duration::from_secs(3));
  targets = msm_benchmarks
}
criterion_main!(msm_gpu_group);

// ---------------------------------------------------------
// Data preparation
// ---------------------------------------------------------
fn prepare_data<F: PrimeField, A: CurveAffine<ScalarExt = F>, S: Send + From<u64>>(
  n: usize,
  bit_width: u32,
) -> (Vec<A>, Vec<S>) {
  let bases: Vec<A> = (0..n)
    .into_par_iter()
    .map_init(
      || OsRng,
      |rng, _| {
        let scalar = F::random(rng);
        A::from(A::Curve::generator() * scalar)
      },
    )
    .collect();

  let mask = (1u64 << bit_width) - 1;
  let coeffs: Vec<S> = (0..n)
    .into_par_iter()
    .map_init(|| OsRng, |_, _| S::from(rand::random::<u64>() & mask))
    .collect();

  (bases, coeffs)
}

// ---------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------
fn msm_benchmark<F: PrimeField, A: CurveAffine<ScalarExt = F>>(name: &str, c: &mut Criterion) {
  let sizes = [
    //1024,
    1024 * 16,
    1024 * 32,
    1024 * 64,
    1024 * 128,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
  ];

  let mut group = c.benchmark_group(format!("MSM-{}", name));

  for bit_width in [1] {
    for &n in &sizes {
      println!("Preparing data for {} elements...", n);
      let (bases, coeffs) = prepare_data::<F, A, u64>(n, bit_width);

      // CPU benchmark
      group.bench_with_input(
        BenchmarkId::new(format!("{}-CPU", name), n),
        &n,
        |b, &_size| {
          b.iter(|| {
            let _ = msm::msm_small(&coeffs, &bases);
          });
        },
      );

      #[cfg(feature = "gpu")]
      // GPU benchmark
      gpu_host::cuda_ctx(0, |ctx, m| {
        let half_len = (bases.len() + 1) / 2;
        let block_dim = (half_len as u32).min(msm_gpu::MAX_BLOCK_DIM);
        let grid_size = (half_len as u32 + block_dim - 1) / block_dim;

        // Allocate GPU buffers
        let d_bases = ctx.new_tensor_view(bases.as_slice()).unwrap();
        let mut d_partial_sums = ctx
          .new_tensor_view(vec![A::Curve::identity(); (grid_size * 2) as usize].as_slice())
          .unwrap();
        let mut d_scalars = ctx.new_tensor_view(coeffs.as_slice()).unwrap();
        group.bench_with_input(
          BenchmarkId::new(format!("n-{}-GPU", name), n),
          &n,
          |b, &_size| {
            b.iter(|| {
              d_scalars.copy_from_host(coeffs.as_slice()).unwrap();
              let _ = msm_gpu::msm_binary_gpu(ctx, m, &d_scalars, &d_bases, &mut d_partial_sums);
            });
          },
        );
      });
    }
  }

  /*
  for bit_width in [16, 32, 48, 63] {
    for &n in &sizes {
      println!("Preparing data for {} elements...", n);
      let (bases, coeffs) = prepare_data::<F, A, F>(n, bit_width);

      // CPU benchmark
      group.bench_with_input(
        BenchmarkId::new(format!("{}-CPU", name), n),
        &n,
        |b, &_size| {
          b.iter(|| {
            let _ = msm::msm(&coeffs, &bases);
          });
        },
      );

      #[cfg(feature = "gpu")]
      // GPU benchmark
      gpu_host::cuda_ctx(0, |ctx, m| {
        let half_len = (bases.len() + 1) / 2;
        let block_dim = (half_len as u32).min(msm_gpu::MAX_BLOCK_DIM);
        let grid_size = (half_len as u32 + block_dim - 1) / block_dim;

        // Allocate GPU buffers
        let d_bases = ctx.new_tensor_view(bases.as_slice()).unwrap();
        let mut d_partial_sums = ctx
          .new_tensor_view(vec![A::Curve::identity(); (grid_size * 2) as usize].as_slice())
          .unwrap();
        let mut d_scalars = ctx.new_tensor_view(coeffs.as_slice()).unwrap();
        group.bench_with_input(
          BenchmarkId::new(format!("n-{}-GPU", name), n),
          &n,
          |b, &_size| {
            b.iter(|| {
              d_scalars.copy_from_host(coeffs.as_slice()).unwrap();
              let _ = msm_gpu::msm_gpu_inner(ctx, m, &d_scalars, &d_bases, &mut d_partial_sums);
            });
          },
        );
      });
    }
  }
  */

  group.finish();
}

#[cfg(feature = "blitzar")]
fn blitzar_benchmark(c: &mut Criterion) {
  let name = "bn256";
  let sizes = [
    1024,
    1024 * 16,
    1024 * 32,
    1024 * 64,
    1024 * 128,
    1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
  ];
  let bit_width = 1;

  let mut group = c.benchmark_group(format!("blitzar-{}", name));

  for &n in &sizes {
    let (bases, coeffs) =
      prepare_data::<halo2curves::bn256::Fr, halo2curves::bn256::G1Affine, halo2curves::bn256::Fr>(
        n, bit_width,
      );
    let coeffs4 = [coeffs.clone(), coeffs.clone(), coeffs.clone(), coeffs];
    let coeffs1 = &coeffs4[0..1];
    // blitzar GPU benchmark
    group.bench_with_input(
      BenchmarkId::new(format!("{}-blitzar-GPU", name), n),
      &n,
      |b, &_size| {
        b.iter(|| {
          let _ = blitzar::batch_vartime_multiscalar_mul(coeffs1, &bases);
        });
      },
    );
    group.bench_with_input(
      BenchmarkId::new(format!("{}-blitzar-GPU-4", name), n),
      &n,
      |b, &_size| {
        b.iter(|| {
          let _ = blitzar::batch_vartime_multiscalar_mul(&coeffs4, &bases);
        });
      },
    );
  }

  group.finish();
}

// ---------------------------------------------------------
// Entry point for Criterion
// ---------------------------------------------------------
fn msm_benchmarks(c: &mut Criterion) {
  #[cfg(feature = "blitzar")]
  blitzar_benchmark(c);
  msm_benchmark::<vesta::Scalar, vesta::Affine>("vesta", c);
  msm_benchmark::<bn256::Scalar, bn256::Affine>("bn256", c);
  msm_benchmark::<grumpkin::Scalar, grumpkin::Affine>("grumpkin", c);
  msm_benchmark::<secp256k1::Scalar, secp256k1::Affine>("secp256k1", c);
  msm_benchmark::<secq256k1::Scalar, secq256k1::Affine>("secq256k1", c);
}
