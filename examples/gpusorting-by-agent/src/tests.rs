extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::*;
use gpu_host::cuda_ctx;
use rand::Rng;

fn run_sort(input: &[u32]) -> Vec<u32> {
    let n = input.len() as u32;
    let thread_blocks = (n + PART_SIZE - 1) / PART_SIZE;
    let hist_size = (RADIX * thread_blocks) as usize;

    let mut result = input.to_vec();
    let mut alt = vec![0u32; n as usize];
    let mut global_hist = vec![0u32; (RADIX * RADIX_PASSES) as usize];
    let mut pass_hist = vec![0u32; hist_size];

    cuda_ctx(0, |ctx, m| {
        let mut d_sort = ctx.new_tensor_view(result.as_mut_slice()).expect("alloc sort");
        let mut d_alt = ctx.new_tensor_view(alt.as_mut_slice()).expect("alloc alt");
        let mut d_global = ctx.new_tensor_view(global_hist.as_mut_slice()).expect("alloc ghist");
        let mut d_pass = ctx.new_tensor_view(pass_hist.as_mut_slice()).expect("alloc phist");

        dispatch_radix_sort(ctx, m, &mut d_sort, &mut d_alt, &mut d_global, &mut d_pass, n);

        d_sort.copy_to_host(&mut result).expect("copy back");
    });

    result
}

fn is_sorted(data: &[u32]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

fn is_permutation(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort();
    sb.sort();
    sa == sb
}

#[test]
fn test_sort_small_random() {
    let mut rng = rand::rng();
    let input: Vec<u32> = (0..PART_SIZE).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Result is not sorted");
    assert!(is_permutation(&input, &result), "Not a permutation");
}

#[test]
fn test_sort_already_sorted() {
    let input: Vec<u32> = (0..PART_SIZE).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result));
    assert_eq!(result, input);
}

#[test]
fn test_sort_reverse() {
    let input: Vec<u32> = (0..PART_SIZE).rev().collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result));
}

#[test]
fn test_sort_all_same() {
    let input = vec![42u32; PART_SIZE as usize];
    let result = run_sort(&input);
    assert!(is_sorted(&result));
    assert_eq!(result, input);
}

#[test]
fn test_sort_non_multiple_size() {
    let mut rng = rand::rng();
    let n = PART_SIZE + 123;
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Non-multiple: not sorted");
    assert!(is_permutation(&input, &result), "Non-multiple: not permutation");
}

#[test]
fn test_sort_two_partitions() {
    let mut rng = rand::rng();
    let n = PART_SIZE * 2;
    let input: Vec<u32> = (0..n).map(|_| rng.random::<u32>()).collect();
    let result = run_sort(&input);
    assert!(is_sorted(&result), "Two partitions: not sorted");
    assert!(is_permutation(&input, &result), "Two partitions: not permutation");
}
