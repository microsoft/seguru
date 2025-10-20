use rand::Rng;

/// Due to CPU and GPU precision issue,
/// we use this function to compare two f32 slices.
pub fn f32_eq(a: &[f32], b: &[f32], eps: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > eps {
            println!(
                "f32 not equal at index {}: a={} b={} diff={}",
                i,
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
            return false;
        }
    }
    true
}

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn random_f32_vec(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<f32>()).collect()
}

#[allow(dead_code)]
/// Returns a Vec of `n` random f32 numbers in [0.0, 1.0)
pub fn random_float4_vec(n: usize) -> Vec<gpu::Float4> {
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
pub fn random_i32_vec(n: usize) -> Vec<i32> {
    let mut rng = rand::rng();
    (0..n).map(|_| rng.random::<i32>()).collect()
}
