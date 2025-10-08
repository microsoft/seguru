mod common;

use common::f32_eq;
use gpu_host::cuda_ctx;

use crate::common::random_f32_vec;
/*pub fn matmul_forward_kernel4(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    C: i32,
    OC: i32,
)
*/

const SQRT_BDIM: u32 = 16;

fn cpu_matmul(inp: &[f32], weight: &[f32], bias: &[f32], c_size: i32, oc_size: i32) -> Vec<f32> {
    let mut out = vec![0.0; inp.len() / c_size as usize * oc_size as usize];
    let m_size = out.len() as i32 / oc_size;
    for m in 0..m_size {
        for oc in 0..oc_size {
            let mut sum = bias[oc as usize];
            for c in 0..c_size {
                sum += inp[(m * c_size + c) as usize] * weight[(oc * c_size + c) as usize];
            }
            out[(m * oc_size + oc) as usize] = sum;
        }
    }
    out
}

/*
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
*/

#[allow(clippy::too_many_arguments)]
fn matmul_forward(
    out: &mut [f32],
    inp: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: u32,
    t_size: u32,
    c_size: u32,
    oc_size: u32,
) {
    /*
     int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
    */

    let gdim_x = (batch * t_size).div_ceil(8 * SQRT_BDIM);
    let gdim_y = (oc_size).div_ceil(8 * SQRT_BDIM);
    let out_len = (batch * t_size * oc_size) as usize;
    let in_len = (batch * t_size * c_size) as usize;
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    assert!(out.len() == out_len);
    assert!(inp.len() == in_len);
    println!(
        "gdim_x {}, gdim_y {}, B {}, T {}, C {}, OC {}",
        gdim_x, gdim_y, batch, t_size, c_size, oc_size
    );
    //assert!(OC == 8 * gdim_y as usize * SQRT_BDIM);
    cuda_ctx(0, |ctx, m| {
        let d_inp = ctx.new_tensor_view(inp).unwrap();
        let mut d_outp = ctx.new_tensor_view(out).unwrap();
        let d_weight = ctx.new_tensor_view(weight).unwrap();
        let d_bias = ctx.new_tensor_view(bias).unwrap();
        let config =
            gpu_host::gpu_config!(gdim_x, gdim_y, 1, @const SQRT_BDIM, @const SQRT_BDIM, 1, 0);
        llm_rs_gpu::matmul_forward_kernel4::launch(
            config,
            ctx,
            m,
            &mut d_outp,
            &d_inp,
            &d_weight,
            &d_bias,
            c_size,
            oc_size,
        )
        .expect("launch failed");
        d_outp.copy_to_host(out).unwrap();
    });
}

#[test]
fn test_matmul() {
    const B: u32 = 1;
    const T: u32 = 128;
    const C: u32 = 32;
    const OC: u32 = 4 * C;
    let h_dinp = random_f32_vec((B * T * C) as usize);
    let h_weight = random_f32_vec((OC * C) as usize);
    let h_bias = random_f32_vec((OC) as usize);
    let mut h_doutp = random_f32_vec((B * T * OC) as usize);
    matmul_forward(&mut h_doutp, &h_dinp, &h_weight, &h_bias, B, T, C, OC);
    let expected = cpu_matmul(&h_dinp, &h_weight, &h_bias, C as i32, OC as i32);

    // CPU and GPU result may differ a bit due to precision issue
    assert!(
        f32_eq(&expected, &h_doutp, 1e-5),
        "expected {:?} \n\n got {:?}",
        &expected[0..64],
        &h_doutp[0..64]
    );
}

#[test]
fn test_matmul2() {
    const B: u32 = 12;
    const T: u32 = 1024;
    const C: u32 = 32;
    const OC: u32 = 4 * C;
    let h_dinp = random_f32_vec((B * T * C) as usize);
    let h_weight = random_f32_vec((OC * C) as usize);
    let h_bias = random_f32_vec((OC) as usize);
    let mut h_doutp = random_f32_vec((B * T * OC) as usize);
    matmul_forward(&mut h_doutp, &h_dinp, &h_weight, &h_bias, B, T, C, OC);
    let expected = cpu_matmul(&h_dinp, &h_weight, &h_bias, C as i32, OC as i32);

    // CPU and GPU result may differ a bit due to precision issue
    assert!(
        f32_eq(&expected, &h_doutp, 1e-5),
        "expected {:?} \n\n got {:?}",
        &expected[0..64],
        &h_doutp[0..64]
    );
}

fn matmul_backward_bias_cpu(
    dout: &[f32], // length B * T * OC
    b: u32,
    t: u32,
    oc: u32,
) -> Vec<f32> {
    let b = b as usize;
    let t = t as usize;
    let oc = oc as usize;
    let mut dbias = vec![0.0; oc];
    assert_eq!(dout.len(), b * t * oc);

    for col in 0..oc {
        let mut sum = 0.0f32;
        for row in 0..(b * t) {
            sum += dout[row * oc + col];
        }
        dbias[col] += sum;
    }
    dbias
}

fn matmul_backward_bias(
    dout: &[f32], // length B * T * OC
    dbias: &mut [f32],
    b: u32,
    t: u32,
    oc: u32,
    epsilon: f32,
) {
    const BSIZE: u32 = 256;
    const SHARED_SIZE: u32 = BSIZE * size_of::<f32>() as u32;
    let gdim_x = oc.div_ceil(32);
    let out_len = (b * t * oc) as usize;
    assert!(dout.len() == out_len);
    assert!(dbias.len() == oc as usize);
    let expected = matmul_backward_bias_cpu(dout, b, t, oc);
    cuda_ctx(0, |ctx, m| {
        let d_dout = ctx.new_tensor_view(dout).unwrap();
        let mut d_dbias = ctx.new_tensor_view(dbias).unwrap();

        let config = gpu_host::gpu_config!(gdim_x, 1, 1, @const BSIZE, 1, 1, @const SHARED_SIZE);
        llm_rs_gpu::matmul_backward_bias_kernel4::launch(
            config,
            ctx,
            m,
            &mut d_dbias,
            &d_dout,
            b,
            t,
            oc,
        )
        .expect("launch failed");
        d_dbias.copy_to_host(dbias).unwrap();
    });
    assert!(
        f32_eq(dbias, &expected, epsilon),
        "dbias not match:\n\n{:?}\n\n{:?}",
        &dbias[0..32.min(oc as usize)],
        &expected[0..32.min(oc as usize)]
    );
}

#[test]
fn test_matmul_backward_bias() {
    const B: u32 = 2;
    const T: u32 = 8;
    const OC: u32 = 32;
    let h_dout = random_f32_vec((B * T * OC) as usize);
    let mut h_dbias = vec![0.0; OC as usize];
    matmul_backward_bias(&h_dout, &mut h_dbias, B, T, OC, 1e-5);
}

#[test]
fn test_matmul_backward_bias_large() {
    const B: u32 = 4;
    const T: u32 = 1024;
    const OC: u32 = 256;
    let h_dout = random_f32_vec((B * T * OC) as usize);
    let mut h_dbias = vec![0.0; OC as usize];
    matmul_backward_bias(&h_dout, &mut h_dbias, B, T, OC, 1e-2); // float precision issue
}
