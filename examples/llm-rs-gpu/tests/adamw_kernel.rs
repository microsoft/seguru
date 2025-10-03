mod common;
use gpu_host::cuda_ctx;

use crate::common::f32_eq;
use crate::common::random_f32_vec;

#[inline(always)]
fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    weight.mul_add(end, (-weight).mul_add(start, start))
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_kernel2_cpu(
    params_memory: &[f32],
    grads_memory: &[f32],
    m_memory: &[f32],
    v_memory: &[f32],
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    beta1_correction: f32,
    beta2_correction: f32,
    eps: f32,
    weight_decay: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(params_memory.len(), grads_memory.len());
    assert_eq!(params_memory.len(), m_memory.len());
    assert_eq!(params_memory.len(), v_memory.len());

    let num_parameters = params_memory.len();

    let mut new_params = Vec::with_capacity(num_parameters);
    let mut new_m = Vec::with_capacity(num_parameters);
    let mut new_v = Vec::with_capacity(num_parameters);

    for i in 0..num_parameters {
        let grad = grads_memory[i];
        let mut m = m_memory[i];
        let mut v = v_memory[i];

        // update first moment
        m = lerp(grad, m, beta1);

        // update second moment
        v = lerp(grad * grad, v, beta2);

        // bias-corrected moments
        let m_hat = m / beta1_correction;
        let v_hat = v / beta2_correction;

        // update parameter
        let param = params_memory[i]
            - learning_rate * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params_memory[i]);

        new_params.push(param);
        new_m.push(m);
        new_v.push(v);
    }

    (new_params, new_m, new_v)
}

#[test]
fn test_adam() {
    let num_params = 1024 * 1024;
    let params = random_f32_vec(num_params);
    let grads = random_f32_vec(num_params);
    let m_memory = random_f32_vec(num_params);
    let v_memory = random_f32_vec(num_params);
    let learning_rate = 3e-4;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let beta1_correction = 1.0 - beta1;
    let beta2_correction = 1.0 - beta2;
    let eps = 1e-8;
    let weight_decay = 1e-2;

    let (new_params_cpu, new_m_cpu, new_v_cpu) = adamw_kernel2_cpu(
        &params,
        &grads,
        &m_memory,
        &v_memory,
        learning_rate,
        beta1,
        beta2,
        beta1_correction,
        beta2_correction,
        eps,
        weight_decay,
    );

    let mut new_params_gpu = vec![0.0f32; num_params];
    let mut new_m_gpu = vec![0.0f32; num_params];
    let mut new_v_gpu = vec![0.0f32; num_params];

    const BSIZE: usize = 512;
    let grid_size = num_params.div_ceil(BSIZE);

    cuda_ctx(0, |ctx, m| {
        let mut d_params = ctx
            .new_tensor_view(params.as_slice())
            .expect("alloc failed");
        let d_grads = ctx.new_tensor_view(grads.as_slice()).expect("alloc failed");
        let mut d_m = ctx
            .new_tensor_view(m_memory.as_slice())
            .expect("alloc failed");
        let mut d_v = ctx
            .new_tensor_view(v_memory.as_slice())
            .expect("alloc failed");
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
        llm_rs_gpu::adamw_kernel2::launch(
            config,
            ctx,
            m,
            &mut d_params,
            &d_grads,
            &mut d_m,
            &mut d_v,
            num_params as _,
            learning_rate,
            beta1,
            beta2,
            beta1_correction,
            beta2_correction,
            eps,
            weight_decay,
        )
        .expect("Failed to run adamw_kernel2");
        d_params
            .copy_to_host(&mut new_params_gpu)
            .expect("copy to host failed");
        d_m.copy_to_host(&mut new_m_gpu)
            .expect("copy to host failed");
        d_v.copy_to_host(&mut new_v_gpu)
            .expect("copy to host failed");
    });
    assert!(
        f32_eq(&new_params_cpu, &new_params_gpu, 1e-5),
        "params not match"
    );
    assert!(f32_eq(&new_m_cpu, &new_m_gpu, 1e-5), "m not match");
    assert!(f32_eq(&new_v_cpu, &new_v_gpu, 1e-5), "v not match");
}
