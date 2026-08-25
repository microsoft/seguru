use crate::driver::radix_sort;

fn check(mut input: Vec<u32>) {
    let mut expected = input.clone();
    expected.sort_unstable();
    let got = radix_sort(&input);
    input.clear();
    assert_eq!(got.len(), expected.len());
    assert!(got == expected, "sorted output mismatch");
}

/// Small deterministic LCG so the tests need no rand dependency at runtime.
fn lcg(seed: u32, n: usize) -> Vec<u32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s
        })
        .collect()
}

#[test]
fn sort_reversed_small() {
    check((0..64u32).rev().collect());
}

#[test]
fn sort_already_sorted() {
    check((0..1024u32).collect());
}

#[test]
fn sort_all_equal() {
    check(vec![7u32; 5000]);
}

#[test]
fn sort_with_extremes() {
    let mut v = lcg(3, 4096);
    v[0] = u32::MAX;
    v[1] = 0;
    v[4095] = u32::MAX;
    check(v);
}

#[test]
fn sort_ragged_not_multiple_of_partition() {
    // 7680 keys per partition; 12345 is deliberately ragged.
    check(lcg(11, 12_345));
}

#[test]
fn sort_multi_partition() {
    check(lcg(29, 1 << 20));
}

#[test]
fn sort_large() {
    check(lcg(97, 1 << 24));
}

#[test]
fn sort_high_bits_only() {
    // Exercises the upper radix passes: low 16 bits are constant.
    check(lcg(5, 20_000).into_iter().map(|x| x & 0xFFFF_0000).collect());
}

#[test]
fn upsweep_and_scan_match_cpu() {
    use crate::*;
    use gpu_host::gpu_config;

    let n = 20_000usize;
    let keys = lcg(7, n);
    let packed = pack_padded(&keys);
    let padded = packed.len() * 4;
    let tb = thread_blocks(n);
    let ptb = padded_thread_blocks(n);
    let shift = 0u32;

    // CPU model
    let mut cpu_pass = vec![0u32; (RADIX * ptb) as usize];
    for b in 0..tb {
        for i in 0..PART_SIZE {
            let idx = (b * PART_SIZE + i) as usize;
            let key = packed[idx / 4].data()[idx % 4];
            let d = (key >> shift) & RADIX_MASK;
            cpu_pass[(d * ptb + b) as usize] += 1;
        }
    }
    let mut cpu_global = vec![0u32; RADIX as usize];
    let mut running = 0u32;
    for d in 0..RADIX as usize {
        cpu_global[d] = running;
        for b in 0..tb {
            running += cpu_pass[(d as u32 * ptb + b) as usize];
        }
    }
    // exclusive scan of pass hist along blocks
    let mut cpu_scan = vec![0u32; (RADIX * ptb) as usize];
    for d in 0..RADIX {
        let mut acc = 0u32;
        for b in 0..ptb {
            cpu_scan[(d * ptb + b) as usize] = acc;
            acc += cpu_pass[(d * ptb + b) as usize];
        }
    }

    gpu_host::cuda_ctx(0, |ctx, m| {
        let zeros_gh = vec![0u32; (RADIX * RADIX_PASSES) as usize];
        let zeros_ph = vec![0u32; (RADIX * ptb) as usize];
        let d_a = ctx.new_tensor_view::<[U32_4]>(&packed).unwrap();
        let mut d_gh = ctx.new_tensor_view::<[u32]>(&zeros_gh).unwrap();
        let mut d_ph = ctx.new_tensor_view::<[u32]>(&zeros_ph).unwrap();
        let up_cfg = gpu_config!(tb, 1, 1, @const UPSWEEP_THREADS, 1, 1, RADIX * 2 * 4);
        upsweep::radix_upsweep::launch(up_cfg, ctx, m, &d_a, &mut d_gh, &mut d_ph, shift, ptb)
            .unwrap();
        ctx.sync().unwrap();
        let mut gh = vec![0u32; (RADIX * RADIX_PASSES) as usize];
        let mut ph = vec![0u32; (RADIX * ptb) as usize];
        d_gh.copy_to_host(&mut gh).unwrap();
        d_ph.copy_to_host(&mut ph).unwrap();
        assert_eq!(padded as u32, tb * PART_SIZE);
        for d in 0..RADIX as usize {
            assert_eq!(gh[d], cpu_global[d], "global_hist digit {d}");
        }
        assert_eq!(ph, cpu_pass, "pass_hist");

        let scan_cfg = gpu_config!(RADIX, 1, 1, @const SCAN_THREADS, 1, 1, SCAN_THREADS * 4);
        scan::radix_scan::launch(scan_cfg, ctx, m, &mut d_ph, ptb).unwrap();
        ctx.sync().unwrap();
        d_ph.copy_to_host(&mut ph).unwrap();
        assert_eq!(ph, cpu_scan, "scanned pass_hist");
    });
}
