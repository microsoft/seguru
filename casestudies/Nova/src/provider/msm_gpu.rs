//! This module provides a multi-scalar multiplication routine
//! The generic implementation is adapted from halo2; we add an optimization to commit to bits more efficiently
//! The specialized implementations are adapted from jolt, with additional optimizations and parallelization.
//! A GPU implementation is also provided for binary scalars.
use ff::{Field, PrimeField};
use gpu_host::*;
use halo2curves::{group::Group, CurveAffine};
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rayon::{current_num_threads, prelude::*};

/// MAX_BLOCK_DIM which is smaller than 1024 to fit smem_size in shared memory
pub const MAX_BLOCK_DIM: u32 = 256;

mod rs_gpu {
  #[allow(unused_imports)]
  use gpu::cg::*;
  #[allow(unused_imports)]
  use gpu::prelude::*;
  use gpu::reshape_map;
  use gpu::sync_threads;
  use halo2curves::group::Group;
  use halo2curves::CurveAffine;
  use num_integer::Integer;
  use num_traits::ToPrimitive;

  #[gpu::cuda_kernel(dynamic_shared)]
  pub fn msm_binary_kernel<C: CurveAffine, T: Integer + Sync + ToPrimitive + 'static>(
    scalars: &[T],
    bases: &[C],
    partial_sums: &mut [C::Curve],
  ) {
    let tid = thread_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let grid_size = block_dim * grid_dim * 2;
    let id = tid + block_dim * block_id::<DimX>();
    let smem = smem_alloc.alloc::<C::Curve>(block_dim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut partial_sums_chunk = chunk_mut(
      partial_sums,
      reshape_map!([1] | [(block_dim, 1), grid_dim] => layout: [i0, t1, t0]),
    );
    let get_data = |idx: u32| {
      if (idx as usize) < scalars.len() && !scalars[idx as usize].is_zero() {
        bases[idx as usize]
      } else {
        C::identity()
      }
    };
    let mut local_sum = C::Curve::identity();
    for i in (id..scalars.len() as u32).step_by(grid_size as usize) {
      local_sum += get_data(i) + get_data(i + grid_size / 2);
    }
    smem_chunk[0] = local_sum;
    sync_threads();

    for order in (0..10).rev() {
      let stride = 1 << order;
      if stride >= block_dim {
        continue;
      }
      let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
      if tid < stride {
        let tmp = smem_chunk[1];
        smem_chunk[0] += tmp;
      }
      sync_threads();
    }
    if tid == 0 {
      partial_sums_chunk[0] = *smem[0];
    }
  }

  #[gpu::cuda_kernel]
  pub fn msm_kernel<C: CurveAffine>(
    scalars: &[C::Scalar],
    bases: &[C],
    partial_sums: &mut [C::Curve],
  ) {
    let mut partial_sums_chunk = chunk_mut(partial_sums, MapLinear::new(1));
    let i = partial_sums_chunk.local2global(0) as usize;
    partial_sums_chunk[0] = bases[i] * scalars[i];
  }

  #[gpu::cuda_kernel(dynamic_shared)]
  pub fn reduce_sum<C, C2>(inputs: &[C], partial_sums: &mut [C2])
  where
    C: core::ops::Add<Output = C2> + Copy + Sync + 'static,
    C2: core::ops::Add<C, Output = C2>
      + core::ops::Add<Output = C2>
      + core::ops::AddAssign
      + Copy
      + Sync
      + 'static
      + Default,
  {
    let tid = thread_id::<DimX>();
    let block_dim = block_dim::<DimX>();
    let id = tid + block_dim * block_id::<DimX>();
    let grid_dim = grid_dim::<DimX>();
    let grid_size = block_dim * grid_dim * 2;
    let smem = smem_alloc.alloc::<C2>(block_dim as usize);
    let mut smem_chunk = smem.chunk_mut(MapLinear::new(1));
    let mut partial_sums_chunk = chunk_mut(
      partial_sums,
      reshape_map!([1] | [(block_dim, 1), grid_dim] => layout: [i0, t1, t0]),
    );

    let mut local_sum = C2::default();
    for i in (id..inputs.len() as u32).step_by(grid_size as usize) {
      local_sum += inputs[i as usize] + inputs[(i + grid_size / 2) as usize];
    }
    smem_chunk[0] = local_sum;
    sync_threads();
    for order in (0..10).rev() {
      let stride = 1 << order;
      if stride >= block_dim {
        continue;
      }
      let mut smem_chunk = smem.chunk_mut(reshape_map!([2] | [stride] => layout: [t0, i0]));
      if tid < stride {
        let tmp = smem_chunk[1];
        smem_chunk[0] += tmp;
      }
      sync_threads();
    }
    if tid == 0 {
      partial_sums_chunk[0] = *smem[0];
    }
  }
}

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
  None,
  Affine(C),
  Projective(C::Curve),
}

impl<C: CurveAffine> Bucket<C> {
  #[allow(dead_code)]
  fn add_assign(&mut self, other: &C) {
    *self = match *self {
      Bucket::None => Bucket::Affine(*other),
      Bucket::Affine(a) => Bucket::Projective(a + *other),
      Bucket::Projective(a) => Bucket::Projective(a + other),
    }
  }

  #[allow(dead_code)]
  fn add(self, other: C::Curve) -> C::Curve {
    match self {
      Bucket::None => other,
      Bucket::Affine(a) => other + a,
      Bucket::Projective(a) => other + a,
    }
  }
}

#[allow(dead_code)]
fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    (f64::from(bases.len() as u32)).ln().ceil() as usize
  };

  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
      return 0;
    }

    let mut v = [0; 8];
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
      *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp %= 1 << c;

    tmp as usize
  }

  let boolean_sum = coeffs
    .iter()
    .zip(bases.iter())
    .filter(|(scalar, _)| *scalar == &C::Scalar::ONE)
    .fold(C::Curve::identity(), |mut acc, (_, base)| {
      acc += *base;
      acc
    });
  let non_boolean_sum = {
    let segments = (256 / c) + 1;
    (0..segments)
      .rev()
      .fold(C::Curve::identity(), |mut acc, segment| {
        (0..c).for_each(|_| acc = acc.double());

        let mut buckets = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
          // skip Booleans
          if *coeff != C::Scalar::ZERO && *coeff != C::Scalar::ONE {
            let coeff = get_at::<C::Scalar>(segment, c, &coeff.to_repr());
            if coeff != 0 {
              buckets[coeff - 1].add_assign(base);
            }
          }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
          running_sum = exp.add(running_sum);
          acc += &running_sum;
        }
        acc
      })
  };

  boolean_sum + non_boolean_sum
}

/// Performs a multi-scalar-multiplication operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
#[allow(dead_code)]
pub fn msm_gpu<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  gpu_host::cuda_ctx(0, |ctx, m| {
    let d_bases = ctx.new_tensor_view(bases).unwrap();
    let mut d_partial_sums = ctx
      .new_tensor_view(vec![C::Curve::identity(); bases.len() * 2].as_slice())
      .unwrap();
    let d_scalars = ctx.new_tensor_view(coeffs).unwrap();
    msm_gpu_inner(ctx, m, &d_scalars, &d_bases, &mut d_partial_sums)
  })
}

#[allow(dead_code)]
fn num_bits(n: usize) -> usize {
  if n == 0 {
    0
  } else {
    (n.ilog2() + 1) as usize
  }
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
#[allow(dead_code)]
pub fn msm_small<C: CurveAffine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive + 'static>(
  scalars: &[T],
  bases: &[C],
) -> C::Curve {
  let max_num_bits = num_bits(scalars.iter().max().unwrap().to_usize().unwrap());
  msm_small_with_max_num_bits(scalars, bases, max_num_bits)
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
#[allow(dead_code)]
pub fn msm_small_with_max_num_bits<
  C: CurveAffine,
  T: Integer + Into<u64> + Copy + Sync + ToPrimitive + 'static,
>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  assert_eq!(bases.len(), scalars.len());

  match max_num_bits {
    0 => C::identity().into(),
    1 => msm_binary(scalars, bases),
    2..=10 => msm_10(scalars, bases, max_num_bits),
    _ => msm_small_rest(scalars, bases, max_num_bits),
  }
}

fn msm_binary<C: CurveAffine, T: Integer + Sync + Copy + ToPrimitive + 'static>(
  scalars: &[T],
  bases: &[C],
) -> C::Curve {
  gpu_host::cuda_ctx(0, |ctx, m| {
    let half_len = bases.len().div_ceil(2) as u32;
    let block_dim: u32 = (half_len).min(MAX_BLOCK_DIM);
    let grid_size = half_len.div_ceil(block_dim);
    let d_bases = ctx.new_tensor_view(bases).unwrap();
    let mut d_partial_sums = ctx
      .new_tensor_view(vec![C::Curve::identity(); (grid_size * 2) as usize].as_slice())
      .unwrap();
    let d_scalars = ctx.new_tensor_view(scalars).unwrap();
    msm_binary_gpu(ctx, m, &d_scalars, &d_bases, &mut d_partial_sums)
  })
}

/// MSM using GPU acceleration.
pub fn msm_binary_gpu<
  'ctx,
  'a,
  C: CurveAffine,
  T: Integer + Sync + Copy + ToPrimitive + 'static,
  N: GpuCtxSpace,
>(
  ctx: &GpuCtxGuard<'ctx, 'a, N>,
  m: &GpuModule<N>,
  d_scalars: &TensorView<'a, [T]>,
  d_bases: &TensorView<'a, [C]>,
  d_partial_sums: &mut TensorViewMut<'a, [C::Curve]>,
) -> C::Curve {
  let half_len = d_bases.len().div_ceil(2) as u32;
  let block_dim: u32 = (half_len).min(MAX_BLOCK_DIM);
  let smem_size = (block_dim + 1) * size_of::<C::Curve>() as u32;
  let mut sum = C::Curve::identity();
  let mut grid_size = half_len.div_ceil(block_dim);
  assert!(d_partial_sums.len() == (half_len.div_ceil(block_dim) * 2) as usize);
  let config = gpu_host::gpu_config!(grid_size, 1, 1, block_dim, 1, 1, smem_size);
  rs_gpu::msm_binary_kernel::launch(config, ctx, m, d_scalars, d_bases, d_partial_sums).unwrap();

  let num_threads = current_num_threads();
  if grid_size as usize <= num_threads {
    let mut cpu_sums = vec![C::Curve::identity(); grid_size as usize];
    let d_sums = d_partial_sums.split_at(grid_size as usize).0;
    d_sums
      .copy_to_host(&mut cpu_sums)
      .expect("copy from device failed");
    sum = cpu_sums
      .par_chunks(1)
      .map(|v| v[0])
      .reduce(C::Curve::identity, |s, evl| s + evl);
  } else {
    let mut ret_offset = 0;
    while grid_size > 1 {
      let half_len = grid_size.div_ceil(2);
      let block_dim: u32 = (half_len).min(MAX_BLOCK_DIM);
      grid_size = (half_len).div_ceil(block_dim);
      ret_offset = (half_len * 2) as usize;
      let (cu_sum, mut next_sum) = d_partial_sums.split_at_mut(ret_offset);
      rs_gpu::reduce_sum::launch(
        gpu_host::gpu_config!(grid_size, 1, 1, block_dim, 1, 1, smem_size),
        ctx,
        m,
        &cu_sum,
        &mut next_sum,
      )
      .expect("reduce_sum kernel launch failed");
    }
    d_partial_sums
      .index(ret_offset)
      .copy_to_host(&mut sum)
      .expect("copy from device failed");
    ctx.sync().unwrap();
  }
  sum
}

/// MSM using GPU acceleration.
pub fn msm_gpu_inner<'ctx, 'a, C: CurveAffine, N: GpuCtxSpace>(
  ctx: &GpuCtxGuard<'ctx, 'a, N>,
  m: &GpuModule<N>,
  d_scalars: &TensorView<'a, [C::Scalar]>,
  d_bases: &TensorView<'a, [C]>,
  d_partial_sums: &mut TensorViewMut<'a, [C::Curve]>,
) -> C::Curve {
  let block_dim: u32 = (d_bases.len() as u32).min(MAX_BLOCK_DIM);
  let mut sum = C::Curve::identity();
  let mut grid_size = (d_bases.len() as u32).div_ceil(block_dim);
  let mut result_size = d_bases.len() as u32;
  assert!(d_partial_sums.len() as u32 == result_size * 2);
  let config = gpu_host::gpu_config!(grid_size, 1, 1, block_dim, 1, 1, 0);
  rs_gpu::msm_kernel::launch(config, ctx, m, d_scalars, d_bases, d_partial_sums).unwrap();
  let num_threads = current_num_threads();
  if result_size as usize <= num_threads && result_size > 1 {
    let mut cpu_sums = vec![C::Curve::identity(); result_size as usize];
    let d_sums = d_partial_sums.split_at(result_size as usize).0;
    d_sums
      .copy_to_host(&mut cpu_sums)
      .expect("copy from device failed");
    sum = cpu_sums
      .par_chunks(1)
      .map(|v| v[0])
      .reduce(C::Curve::identity, |s, evl| s + evl);
  } else {
    let mut ret_offset = 0;
    while result_size > 1 {
      ret_offset = result_size as usize;
      let half_len = result_size.div_ceil(2);
      let block_dim: u32 = (half_len).min(MAX_BLOCK_DIM);
      grid_size = (half_len).div_ceil(block_dim);
      result_size = grid_size;
      let smem_size = (block_dim + 1) * size_of::<C::Curve>() as u32;
      let (cu_sum, mut next_sum) = d_partial_sums.split_at_mut(ret_offset);
      rs_gpu::reduce_sum::launch(
        gpu_host::gpu_config!(grid_size, 1, 1, block_dim, 1, 1, smem_size),
        ctx,
        m,
        &cu_sum,
        &mut next_sum,
      )
      .expect("reduce_sum kernel launch failed");
    }
    d_partial_sums
      .index(ret_offset)
      .copy_to_host(&mut sum)
      .expect("copy from device failed");
    ctx.sync().unwrap();
  }
  sum
}

/// MSM optimized for up to 10-bit scalars
fn msm_10<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_10_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let num_buckets: usize = 1 << max_num_bits;
    let mut buckets = vec![Bucket::None; num_buckets];

    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _base)| !scalar.is_zero())
      .for_each(|(scalar, base)| {
        let bucket_index: u64 = (*scalar).into();
        buckets[bucket_index as usize].add_assign(base);
      });

    let mut result = C::Curve::identity();
    let mut running_sum = C::Curve::identity();
    buckets.iter().skip(1).rev().for_each(|exp| {
      running_sum = exp.add(running_sum);
      result += &running_sum;
    });
    result
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| msm_10_serial(scalars_chunk, bases_chunk, max_num_bits))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_10_serial(scalars, bases, max_num_bits)
  }
}

#[allow(dead_code)]
fn msm_small_rest<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_small_rest_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let mut c = if bases.len() < 32 {
      3
    } else {
      compute_ln(bases.len()) + 2
    };

    if max_num_bits == 32 || max_num_bits == 64 {
      c = 8;
    }

    let zero = C::Curve::identity();

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts = (0..max_num_bits).step_by(c);

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
      .map(|w_start| {
        let mut res = zero;
        // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
        let mut buckets = vec![zero; (1 << c) - 1];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
          let scalar: u64 = scalar.into();
          if scalar == 1 {
            // We only process unit scalars once in the first window.
            if w_start == 0 {
              res += base;
            }
          } else {
            let mut scalar = scalar;

            // We right-shift by w_start, thus getting rid of the
            // lower bits.
            scalar >>= w_start;

            // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
            scalar %= 1 << c;

            // If the scalar is non-zero, we update the corresponding
            // bucket.
            // (Recall that `buckets` doesn't have a zero bucket.)
            if scalar != 0 {
              buckets[(scalar - 1) as usize] += base;
            }
          }
        });

        // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
        // This is computed below for b buckets, using 2b curve additions.
        //
        // We could first normalize `buckets` and then use mixed-addition
        // here, but that's slower for the kinds of groups we care about
        // (Short Weierstrass curves and Twisted Edwards curves).
        // In the case of Short Weierstrass curves,
        // mixed addition saves ~4 field multiplications per addition.
        // However normalization (with the inversion batched) takes ~6
        // field multiplications per element,
        // hence batch normalization is a slowdown.

        // `running_sum` = sum_{j in i..num_buckets} bucket[j],
        // where we iterate backward from i = num_buckets to 0.
        let mut running_sum = C::Curve::identity();
        buckets.into_iter().rev().for_each(|b| {
          running_sum += &b;
          res += &running_sum;
        });
        res
      })
      .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
      + window_sums[1..]
        .iter()
        .rev()
        .fold(zero, |mut total, sum_i| {
          total += sum_i;
          for _ in 0..c {
            total = total.double();
          }
          total
        })
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| {
        msm_small_rest_serial(scalars_chunk, bases_chunk, max_num_bits)
      })
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_small_rest_serial(scalars, bases, max_num_bits)
  }
}

fn compute_ln(a: usize) -> usize {
  // log2(a) * ln(2)
  if a == 0 {
    0 // Handle edge case where log2 is undefined
  } else {
    a.ilog2() as usize * 69 / 100
  }
}

/// cargo test --no-default-features --features rs_gpu
#[cfg(test)]
mod tests {
  use super::*;
  /*use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    pasta::{pallas, vesta},
    secp_secq::{secp256k1, secq256k1},
  };*/
  use crate::provider::bn256_grumpkin::bn256;
  use halo2curves::CurveAffine;
  use rand_core::OsRng;

  fn test_msm_ux_with<F: PrimeField, A: CurveAffine<ScalarExt = F>>() {
    let n = 1;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    //.map(|_| rand::random::<u64>() 1% (1 << bit_width))
    for bit_width in [1] {
      let coeffs: Vec<u64> = (0..n).map(|_| 1 % (1 << bit_width)).collect::<Vec<_>>();

      let coeffs_scalar: Vec<F> = coeffs.iter().map(|b| F::from(*b)).collect::<Vec<_>>();
      let general_cpu = crate::provider::msm::msm(&coeffs_scalar, &bases);
      let general_gpu = msm_gpu(&coeffs_scalar, &bases);
      let integer = msm_binary(&coeffs, &bases);
      assert_eq!(general_cpu, integer);
      assert_eq!(general_gpu, general_cpu);
    }
  }

  #[test]
  fn test_gpu_msm_ux() {
    test_msm_ux_with::<bn256::Scalar, bn256::Affine>();
    /*test_msm_ux_with::<pallas::Scalar, pallas::Affine>();
    test_msm_ux_with::<vesta::Scalar, vesta::Affine>();
    test_msm_ux_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_ux_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_ux_with::<secq256k1::Scalar, secq256k1::Affine>();*/
  }
}
