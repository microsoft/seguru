use cublas_sys::{cublasSgemm_v2, cublasSgemmStridedBatched};
use cudarc::cublas::sys as cublas_sys;
use gpu::Float4;
use gpu_host::{GpuCtxGuard, GpuCtxSpace, GpuModule, TensorView, TensorViewMut};
use llm_rs_gpu::*;
use log::*;

struct GPUExecContext<'ctx, 'a, CN: GpuCtxSpace> {
    pub ctx: GpuCtxGuard<'ctx, 'a, CN>,
    pub m: GpuModule<CN>,
    pub cublas_handle: cublas_sys::cublasHandle_t,
}

pub fn encoder_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    inp: &mut TensorViewMut<'_, [i32]>,
    wte: &TensorView<'_, [f32]>,
    wpe: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    assert!(channel % 4 == 0);
    let n = batch_size * seq_len * channel;
    const BSIZE: usize = 512;
    let grid_size = (n / 4).div_ceil(BSIZE);
    let out = unsafe { &mut *(out as *mut _ as *mut TensorViewMut<'_, [Float4]>) };
    let wte = unsafe { &*(wte as *const _ as *const TensorViewMut<'_, [Float4]>) };
    let wpe = unsafe { &*(wpe as *const _ as *const TensorViewMut<'_, [Float4]>) };
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    let start = std::time::Instant::now();
    encoder_forward_kernel3::launch(
        config,
        ctx,
        m,
        out,
        inp,
        wte,
        wpe,
        batch_size as _,
        seq_len as _,
        channel as _,
    )
    .expect("Failed to run encoder_forward_kernel3");
    trace!("encoder_forward: {:?}", start.elapsed());
}

/*
void encoder_backward(float* dwte, float* dwpe,
                    const float* dout, const int* inp,
                    int B, int T, int C) {
    const int N = B * T * C;
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    encoder_backward_kernel<<<grid_size, block_size>>>(dwte, dwpe, dout, inp, B, T, C);
    cudaCheck(cudaGetLastError());
}
*/

pub fn encoder_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dwte: &mut TensorViewMut<'_, [f32]>,
    dwpe: &mut TensorViewMut<'_, [f32]>,
    dout: &TensorView<'_, [f32]>,
    inp: &mut TensorViewMut<'_, [i32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    let n = batch_size * seq_len * channel;
    const BSIZE: usize = 256;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    encoder_backward_kernel::launch(
        config,
        ctx,
        m,
        dwte,
        dwpe,
        dout,
        inp,
        batch_size as _,
        seq_len as _,
        channel as _,
    )
    .expect("failed to launch encoder_backward_kernel");
}

/*
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_forward_kernel3<<<grid_size, block_size>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
*/

pub fn layernorm_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    mean: &mut TensorViewMut<'_, [f32]>,
    rstd: &mut TensorViewMut<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    weight: &TensorView<'_, [f32]>,
    bias: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    let n = batch_size * seq_len;
    const BSIZE: usize = 512;
    let grid_size = (n * 32).div_ceil(BSIZE);
    let len = channel * n;
    assert!(inp.len() == len, "{} != {}", inp.len(), len);
    assert!(out.len() == len, "{} != {}", out.len(), len);
    assert!(mean.len() == n);
    assert!(rstd.len() == n);
    assert!(weight.len() == channel);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    //let start = std::time::Instant::now();
    layernorm_forward_kernel3::launch(
        config,
        ctx,
        m,
        out,
        mean,
        rstd,
        inp,
        weight,
        bias,
        n as _,
        channel as _,
    )
    .expect("failed to launch layernorm_forward_kernel3");
    //println!("layernorm_forward: {:?}", start.elapsed());
}

/*
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        const float* dout, const float* inp, const  float* weight, const float* mean, const float* rstd,
                        int B, int T, int C) {
    const int block_size = 512;
    const int N = B * T;
    const int grid_size = CEIL_DIV(32*N, block_size);
    size_t shared_mem_size = 2 * C * sizeof(float);
    layernorm_backward_kernel2<<<grid_size, block_size, shared_mem_size>>>(dinp, dweight, dbias, dout, inp, weight, mean, rstd, B, T, C);
    cudaCheck(cudaGetLastError());
}
*/

pub fn layernorm_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dinp: &mut TensorViewMut<'_, [f32]>,
    dweight: &mut TensorViewMut<'_, [f32]>,
    dbias: &mut TensorViewMut<'_, [f32]>,
    dout: &TensorView<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    weight: &TensorView<'_, [f32]>,
    mean: &TensorView<'_, [f32]>,
    rstd: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
) {
    const BSIZE: usize = 512;
    let n = batch_size * seq_len;
    let grid_size = (32 * n).div_ceil(BSIZE);
    let shared_mem_size = 2 * channel * std::mem::size_of::<f32>();
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, shared_mem_size as u32);
    layernorm_backward_kernel2::launch(
        config,
        ctx,
        m,
        dinp,
        dweight,
        dbias,
        dout,
        inp,
        weight,
        mean,
        rstd,
        batch_size as _,
        seq_len as _,
        channel as _,
    )
    .expect("failed to launch layernorm_backward_kernel2");
}

/*
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    int sqrt_block_size = 16;

    dim3 gridDim(CEIL_DIV(B * T, 8*sqrt_block_size), CEIL_DIV(OC, 8*sqrt_block_size));
    dim3 blockDim(sqrt_block_size, sqrt_block_size);
    matmul_forward_kernel4<<<gridDim, blockDim>>>(out, inp, weight, bias, C, OC);
    cudaCheck(cudaGetLastError());
}*/

pub fn matmul_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    weight: &TensorView<'_, [f32]>,
    bias: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    out_channel: usize,
) {
    let inp = unsafe { &*(inp as *const _ as *const TensorView<'_, [Float4]>) };
    let out = unsafe { &mut *(out as *mut _ as *mut TensorViewMut<'_, [Float4]>) };
    let weight = unsafe { &*(weight as *const _ as *const TensorView<'_, [Float4]>) };
    let bias = unsafe { &*(bias as *const _ as *const TensorView<'_, [Float4]>) };
    let n = batch_size * seq_len;
    const SQRT_BLOCK_SIZE: usize = 16;
    let grid_x = (n).div_ceil(8 * SQRT_BLOCK_SIZE);
    let grid_y = (out_channel).div_ceil(8 * SQRT_BLOCK_SIZE);
    let config = gpu_host::gpu_config!(grid_x as u32, grid_y as u32, 1, @const SQRT_BLOCK_SIZE as u32, @const SQRT_BLOCK_SIZE as u32, 1, 0);
    let start = std::time::Instant::now();
    matmul_forward_kernel4::launch(
        config,
        ctx,
        m,
        out,
        inp,
        weight,
        bias,
        channel as _,
        out_channel as _,
    )
    .expect("failed to launch matmul_forward_kernel4");
    trace!("matmul_forward: {:?}", start.elapsed());
}

/*
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    float one = 1.0f;
    float zero = 0.0f;
    // backward to input, uses = in the backward pass (set the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B*T, OC, &one, weight, C, dout, OC, &zero, dinp, C));
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, C, OC, B*T, &one, inp, C, dout, OC, &one, dweight, C));
    // backward to bias, if given, does a +=
    if (dbias != NULL) {
        const int block_size = 1024;
        const int grid_size = OC / 32; // for now, OC must be divisible by 32 for this kernel to work
        matmul_backward_bias_kernel4<<<grid_size, block_size, block_size * sizeof(float)>>>(dbias, dout, B, T, OC);
        cudaCheck(cudaGetLastError());
    }
}
*/

pub fn matmul_forward_kernel4<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    weight: &TensorView<'_, [f32]>,
    bias: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    out_channel: usize,
) {
    let inp = unsafe { &*(inp as *const _ as *const TensorView<'_, [Float4]>) };
    let out = unsafe { &mut *(out as *mut _ as *mut TensorViewMut<'_, [Float4]>) };
    let weight = unsafe { &*(weight as *const _ as *const TensorView<'_, [Float4]>) };
    let bias = unsafe { &*(bias as *const _ as *const TensorView<'_, [Float4]>) };
    let n = batch_size * seq_len;
    const SQRT_BLOCK_SIZE: usize = 16;
    let grid_x = (n).div_ceil(8 * SQRT_BLOCK_SIZE);
    let grid_y = (out_channel).div_ceil(8 * SQRT_BLOCK_SIZE);
    let config = gpu_host::gpu_config!(grid_x as u32, grid_y as u32, 1, @const SQRT_BLOCK_SIZE as u32, @const SQRT_BLOCK_SIZE as u32, 1, 0);
    matmul_forward_kernel4::launch(
        config,
        ctx,
        m,
        out,
        inp,
        weight,
        bias,
        channel as _,
        out_channel as _,
    )
    .expect("failed to launch matmul_forward_kernel4");
}

pub fn matmul_backward_bias_kernel4<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dbias: &mut TensorViewMut<'_, [f32]>,
    dout: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    out_channel: usize,
) {
    const BSIZE: usize = 1024;
    const SMEM_SIZE: usize = BSIZE * std::mem::size_of::<f32>();
    let grid_size = out_channel.div_ceil(32);
    let config =
        gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, SMEM_SIZE as u32);
    matmul_backward_bias_kernel4::launch(
        config,
        ctx,
        m,
        dbias,
        dout,
        batch_size as _,
        seq_len as _,
        out_channel as _,
    )
    .expect("failed to launch matmul_backward_bias_kernel4");
}

pub(crate) fn matmul_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    cublas_handle: cublas_sys::cublasHandle_t,
    dinp: &mut TensorViewMut<'_, [f32]>,
    dweight: &mut TensorViewMut<'_, [f32]>,
    dbias: Option<&mut TensorViewMut<'_, [f32]>>,
    dout: &TensorView<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    weight: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    out_channel: usize,
) {
    let n = batch_size * seq_len;
    let one = 1.0f32;
    let zero = 0.0f32;
    // backward to input, uses = in the backward pass (set the gradient)
    unsafe {
        let ret = cublasSgemm_v2(
            cublas_handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            channel as i32,
            (batch_size * seq_len) as i32,
            out_channel as i32,
            &one,
            weight.as_devptr() as _,
            channel as i32,
            dout.as_devptr() as _,
            out_channel as i32,
            &zero,
            dinp.as_devptr() as _,
            channel as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward to weight, uses += in the backward pass (accumulate the gradient)
    unsafe {
        let ret = cublasSgemm_v2(
            cublas_handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N,
            cublas_sys::cublasOperation_t::CUBLAS_OP_T,
            channel as i32,
            out_channel as i32,
            (batch_size * seq_len) as i32,
            &one,
            inp.as_devptr() as _,
            channel as i32,
            dout.as_devptr() as _,
            out_channel as i32,
            &one,
            dweight.as_devptr() as _,
            channel as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward to bias, if given, does a +=
    if let Some(dbias) = dbias {
        const BSIZE: usize = 1024;
        const SMEM_SIZE: usize = BSIZE * std::mem::size_of::<f32>();
        let grid_size = out_channel.div_ceil(32);
        let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, SMEM_SIZE as u32);
        assert!(dbias.len() >= out_channel);
        matmul_backward_bias_kernel4::launch(
            config,
            ctx,
            m,
            dbias,
            dout,
            batch_size as _,
            seq_len as _,
            out_channel as _,
        )
        .expect("failed to launch matmul_backward_bias_kernel4");
    }
}

/*
void attention_forward(float* out, float* qkvr, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // Note: `inp` is not needed for backward pass, so we re-use it as a scratch buffer.
    // Its contents will be overwritten by this function.
    const int block_size = 256;
    const int softmax_block_size = 256;

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    int total_threads = B * NH * T * HS;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    permute_kernel<<<num_blocks, block_size>>>(q, k, v, inp, B, T, NH, HS);
    cudaCheck(cudaGetLastError());

    // batched matrix multiply with cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* preatt = inp;
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));

    // multiply all elements of preatt elementwise by scale
    float scale = 1.0 / sqrtf(HS);
    int grid_size = CEIL_DIV(B * NH * T * 32, softmax_block_size);
    softmax_forward_kernel5<<<grid_size, softmax_block_size>>>(att, scale, preatt, B * NH, T);
    cudaCheck(cudaGetLastError());

    // new approach: first cuBLAS another batched matmul
    float* vaccum = inp;
    // y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));

    // now unpermute
    // y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel<<<num_blocks, block_size>>>(vaccum, out, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn attention_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    cublas_handle: cublas_sys::cublasHandle_t,
    out: &mut TensorViewMut<'_, [f32]>,
    qkvr: &mut TensorViewMut<'_, [f32]>,
    att: &mut TensorViewMut<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    num_heads: usize,
) {
    let start = std::time::Instant::now();
    const BSIZE: usize = 256;
    let head_size = channel / num_heads;
    assert!(channel % num_heads == 0);
    let total_threads = batch_size * num_heads * seq_len * head_size;
    let num_blocks = total_threads.div_ceil(BSIZE);
    let bsc_len = batch_size * seq_len * channel;

    let (mut q, mut rest) = qkvr.split_at_mut(bsc_len);
    let (mut k, mut v) = rest.split_at_mut(bsc_len);
    assert!(q.len() == bsc_len);
    assert!(k.len() == bsc_len);
    assert!(v.len() == bsc_len);
    let _ = qkvr;
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    permute_kernel::launch(
        config,
        ctx,
        m,
        &mut q,
        &mut k,
        &mut v,
        inp,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch permute_kernel");
    // batched matrix multiply with cuBLAS
    const CUBLAS_OP_T: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
    const CUBLAS_OP_N: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let preatt = inp;
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &alpha, k, HS, T * HS, q, HS, T * HS, &beta, preatt, T, T * T, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len as i32,
            seq_len as i32,
            head_size as i32,
            &alpha,
            k.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            q.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            &beta,
            preatt.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // multiply all elements of preatt elementwise by scale
    let scale = 1.0f32 / (head_size as f32).sqrt();
    let grid_size = (batch_size * num_heads * seq_len * 32).div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    let preatt = unsafe { &*(preatt as *const _ as *const TensorView<'_, [Float4]>) };
    softmax_forward_kernel5::launch(
        config,
        ctx,
        m,
        att,
        scale,
        preatt,
        (batch_size * num_heads) as _,
        seq_len as _,
    )
    .expect("failed to launch softmax_forward_kernel5");

    let vaccum = inp;
    // new approach: first cuBLAS another batched matmul
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &alpha, v, HS, T * HS, att, T, T * T, &beta, vaccum, HS, T * HS, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_size as i32,
            seq_len as i32,
            seq_len as i32,
            &alpha,
            v.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            att.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            &beta,
            vaccum.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }

    // now unpermute
    let total_threads = batch_size * seq_len * channel;
    let num_blocks = total_threads.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    unpermute_kernel::launch(
        config,
        ctx,
        m,
        vaccum,
        out,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch unpermute_kernel");
    trace!("attention_forward: {:?}", start.elapsed());
}

/*
void attention_backward(float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
                        const float* dout,
                        const float* qkvr, const float* att,
                        int B, int T, int C, int NH) {
    const int block_size = 256;
    int HS = C / NH; // head size
    const float one = 1.0f;
    const float zero = 0.0f; // note beta = 1.0f so that we accumulate gradients (+=)
    // unpack convenience pointers into q, k, v
    const float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;
    float *dq, *dk, *dv;
    dq = dqkvr + 0 * B * T * C;
    dk = dqkvr + 1 * B * T * C;
    dv = dqkvr + 2 * B * T * C;
    // backward through the unpermute operation
    int num_blocks = CEIL_DIV(B * T * C, block_size);
    unpermute_kernel_backward<<<num_blocks, block_size>>>(scratch, dout, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
    // backward into datt
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
    // backward into dv
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
    // backward into preatt
    int hs = C / NH; // head size
    float scale = 1.0f / sqrtf(hs);
    softmax_autoregressive_backward_kernel<<<dim3(T / 4, B * NH), 256>>>(dpreatt, datt, att, B, T, C, scale);
    cudaCheck(cudaGetLastError());
    // backward into q
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
    // backward into k
    cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
    // backward into inp
    num_blocks = CEIL_DIV(B * NH * T * HS, block_size);
    permute_kernel_backward<<<num_blocks, block_size>>>(dinp, dq, dk, dv, B, T, NH, HS);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn attention_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    cublas_handle: cublas_sys::cublasHandle_t,
    dinp: &mut TensorViewMut<'_, [f32]>,
    dqkvr: &mut TensorViewMut<'_, [f32]>,
    dpreatt: &mut TensorViewMut<'_, [f32]>,
    datt: &mut TensorViewMut<'_, [f32]>,
    scratch: &mut TensorViewMut<'_, [f32]>,
    dout: &TensorView<'_, [f32]>,
    qkvr: &TensorView<'_, [f32]>,
    att: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    num_heads: usize,
) {
    const BSIZE: usize = 256;
    let head_size = channel / num_heads;
    assert!(channel % num_heads == 0);
    let bsc_len = batch_size * seq_len * channel;
    let dqkvr_len = 3 * bsc_len;
    assert!(dqkvr.len() >= dqkvr_len);

    let (q, rest) = qkvr.split_at(bsc_len);
    let (k, v) = rest.split_at(bsc_len);
    let q = qkvr.index(0..bsc_len);

    let (dq, mut rest) = dqkvr.split_at_mut(bsc_len);
    let (dk, dv) = rest.split_at_mut(bsc_len);

    let num_blocks = bsc_len.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    unpermute_kernel_backward::launch(
        config,
        ctx,
        m,
        scratch,
        dout,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch permute_kernel_backward");
    // backward into datt
    const CUBLAS_OP_T: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_T;
    const CUBLAS_OP_N: cublas_sys::cublasOperation_t = cublas_sys::cublasOperation_t::CUBLAS_OP_N;
    let one = 1.0f32;
    let zero = 0.0f32; // note beta = 1.0f so that we accumulate gradients (+=)
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, T, T, HS, &one, v, HS, T * HS, scratch, HS, T * HS, &zero, datt, T, T * T, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq_len as i32,
            seq_len as i32,
            head_size as i32,
            &one,
            v.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            scratch.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            &zero,
            datt.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward into dv
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, scratch, HS, T * HS, att, T, T * T, &zero, dv, HS, T * HS, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            head_size as i32,
            seq_len as i32,
            seq_len as i32,
            &one,
            scratch.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            att.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            &zero,
            dv.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward into preatt
    let scale = 1.0f32 / (head_size as f32).sqrt();
    let config = gpu_host::gpu_config!(
        (seq_len / 4) as u32,
        (batch_size * num_heads) as u32,
        1,
        256,
        1,
        1,
        0
    );
    softmax_autoregressive_backward_kernel::launch(
        config,
        ctx,
        m,
        dpreatt,
        datt,
        att,
        batch_size as _,
        seq_len as _,
        channel as _,
        scale,
    )
    .expect("failed to launch softmax_autoregressive_backward_kernel");
    // backward into q
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, HS, T, T, &one, k, HS, T * HS, dpreatt, T, T * T, &zero, dq, HS, T * HS, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_size as i32,
            seq_len as i32,
            seq_len as i32,
            &one,
            k.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            dpreatt.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            &zero,
            dq.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward into k
    unsafe {
        //cublasCheck(cublasSgemmStridedBatched(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, HS, T, T, &one, q, HS, T * HS, dpreatt, T, T * T, &zero, dk, HS, T * HS, B * NH));
        let ret = cublasSgemmStridedBatched(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            head_size as i32,
            seq_len as i32,
            seq_len as i32,
            &one,
            q.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            dpreatt.as_devptr() as _,
            seq_len as i32,
            (seq_len * seq_len) as i64,
            &zero,
            dk.as_devptr() as _,
            head_size as i32,
            (seq_len * head_size) as i64,
            (batch_size * num_heads) as i32,
        );
        assert!(ret == cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    }
    // backward into inp
    let total_threads = batch_size * num_heads * seq_len * head_size;
    let num_blocks = total_threads.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(num_blocks as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    permute_kernel_backward::launch(
        config,
        ctx,
        m,
        dinp,
        &dq,
        &dk,
        &dv,
        batch_size as _,
        seq_len as _,
        num_heads as _,
        head_size as _,
    )
    .expect("failed to launch permute_kernel_backward");
}

/*
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size);
    residual_forward_kernel<<<grid_size, block_size>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn residual_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    inp1: &TensorView<'_, [f32]>,
    inp2: &TensorView<'_, [f32]>,
    n: usize,
) {
    const BSIZE: usize = 256;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    residual_forward_kernel::launch(config, ctx, m, out, inp1, inp2, n as _)
        .expect("failed to launch residual_forward_kernel");
}

/*
void gelu_forward(float* out, const float* inp, int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_forward_kernel<<<grid_size, block_size>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}*/

pub(crate) fn gelu_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    out: &mut TensorViewMut<'_, [f32]>,
    inp: &TensorView<'_, [f32]>,
    n: usize,
) {
    const BSIZE: usize = 128;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    gelu_forward_kernel::launch(config, ctx, m, out, inp, n as _)
        .expect("failed to launch gelu_forward_kernel");
}

/*
void gelu_backward(float* dinp, const float* inp, const float* dout, const int N) {
    const int block_size = 128;
    const int grid_size = CEIL_DIV(N, block_size);
    gelu_backward_kernel<<<grid_size, block_size>>>(dinp, inp, dout, N);
    cudaCheck(cudaGetLastError());
}
*/

pub(crate) fn gelu_backward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dinp: &mut TensorViewMut<'_, [f32]>,
    dout: &TensorView<'_, [f32]>,
    n: usize,
) {
    const BSIZE: usize = 128;
    let grid_size = n.div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    gelu_backward_kernel::launch(config, ctx, m, dinp, dout, n as _)
        .expect("failed to launch gelu_backward_kernel");
}

/*
void fused_classifier3(float* logits, float* losses,
                      const float* dlosses, const int* targets,
                      int B, int T, int V, int P) {
    const int block_size = 1024;
    const int N = B * T;
    const int grid_size = N;
    fused_classifier_kernel3<<<grid_size, block_size>>>(logits, losses, NULL, dlosses, targets, B, T, V, P);
    cudaCheck(cudaGetLastError());
}
*/

pub fn fused_classifier3<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    logits: &mut TensorViewMut<'_, [f32]>,
    losses: &mut TensorViewMut<'_, [f32]>,
    dlosses: &TensorView<'_, [f32]>,
    targets: &TensorView<'_, [i32]>,
    batch_size: usize,
    seq_len: usize,
    vocab_size: usize,
    pad_vocab_size: usize,
) {
    const BSIZE: usize = 1024;
    let grid_size = batch_size * seq_len;
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    let mut empty_tensor = ctx.new_tensor_view([].as_slice()).unwrap();
    fused_classifier_kernel3::launch(
        config,
        ctx,
        m,
        logits,
        losses,
        &mut empty_tensor,
        dlosses,
        targets,
        batch_size as _,
        seq_len as _,
        vocab_size as _,
        pad_vocab_size as _,
    )
    .expect("failed to launch fused_classifier_kernel3");
}

pub fn softmax_forward<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dpreatt: &mut TensorViewMut<'_, [f32]>,
    att: &TensorView<'_, [Float4]>,
    batch_size: usize,
    seq_len: usize,
    num_heads: usize,
    scale: f32,
) {
    const BSIZE: usize = 256;
    let grid_size = (batch_size * num_heads * seq_len * 32).div_ceil(BSIZE);
    let config = gpu_host::gpu_config!(grid_size as u32, 1, 1, @const BSIZE as u32, 1, 1, 0);
    softmax_forward_kernel5::launch(
        config,
        ctx,
        m,
        dpreatt,
        scale,
        att,
        (batch_size * num_heads) as _,
        seq_len as _,
    )
    .expect("failed to launch softmax_forward_kernel5");
}

pub fn softmax_autoregressive_backward_kernel<'ctx, CN: GpuCtxSpace>(
    ctx: &GpuCtxGuard<'ctx, '_, CN>,
    m: &GpuModule<CN>,
    dpreatt: &mut TensorViewMut<'_, [f32]>,
    datt: &TensorView<'_, [f32]>,
    att: &TensorView<'_, [f32]>,
    batch_size: usize,
    seq_len: usize,
    channel: usize,
    scale: f32,
) {
    let config = gpu_host::gpu_config!((seq_len / 4) as u32, (batch_size) as u32, 1, 256, 1, 1, 0);
    softmax_autoregressive_backward_kernel::launch(
        config,
        ctx,
        m,
        dpreatt,
        datt,
        att,
        batch_size as _,
        seq_len as _,
        channel as _,
        scale,
    )
    .expect("failed to launch softmax_autoregressive_backward_kernel");
}
