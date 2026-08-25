//! Safe wrappers around the CUDA C++ reference kernels in
//! `cuda/polybench_ref.cu`.
//!
//! This is the only module in the crate that contains `unsafe`, and it exists
//! solely so the benchmark can compare against hand-written CUDA. The SeGuRu
//! kernels and their host code are entirely safe Rust.
//!
//! Every entry point takes host slices, runs the CUDA reference for
//! `warmup + iters` iterations on device-resident buffers, and returns the mean
//! *kernel-only* time in microseconds together with the output of one clean
//! (re-initialised) run for verification.

/// Mean kernel time in microseconds plus the reference output.
pub type Timed<T> = (f64, T);

unsafe extern "C" {
    fn cuda_gemm_bench(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        nk: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_gemm_cublas_bench(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        nk: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_twomm_bench(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        d: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        nk: i32,
        nl: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_threemm_bench(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        d: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        nk: i32,
        nl: i32,
        nm: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_syrk_bench(
        a: *const f32,
        c: *const f32,
        out: *mut f32,
        n: i32,
        m: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_syr2k_bench(
        a: *const f32,
        b: *const f32,
        c: *const f32,
        out: *mut f32,
        n: i32,
        m: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_atax_bench(
        a: *const f32,
        x: *const f32,
        out: *mut f32,
        nx: i32,
        ny: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_bicg_bench(
        a: *const f32,
        p: *const f32,
        r: *const f32,
        s: *mut f32,
        q: *mut f32,
        nx: i32,
        ny: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_gesummv_bench(
        a: *const f32,
        b: *const f32,
        x: *const f32,
        out: *mut f32,
        n: i32,
        alpha: f32,
        beta: f32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_mvt_bench(
        a: *const f32,
        x1: *const f32,
        x2: *const f32,
        y1: *const f32,
        y2: *const f32,
        ox1: *mut f32,
        ox2: *mut f32,
        n: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_conv2d_bench(
        a: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_conv3d_bench(
        a: *const f32,
        out: *mut f32,
        ni: i32,
        nj: i32,
        nk: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_jacobi1d_bench(
        a: *const f32,
        b: *const f32,
        oa: *mut f32,
        ob: *mut f32,
        n: i32,
        tsteps: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_jacobi2d_bench(
        a: *const f32,
        b: *const f32,
        oa: *mut f32,
        ob: *mut f32,
        n: i32,
        tsteps: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
    fn cuda_fdtd2d_bench(
        ex: *const f32,
        ey: *const f32,
        hz: *const f32,
        fict: *const f32,
        oex: *mut f32,
        oey: *mut f32,
        ohz: *mut f32,
        nx: i32,
        ny: i32,
        tmax: i32,
        warmup: i32,
        iters: i32,
    ) -> f32;
}

fn us(ms: f32) -> f64 {
    ms as f64 * 1000.0
}

/// `C = alpha * A * B + beta * C`, mirroring [`crate::gemm::gemm_kernel`].
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nk);
    assert_eq!(b.len(), nk * nj);
    assert_eq!(c.len(), ni * nj);
    let mut out = vec![0.0f32; ni * nj];
    // SAFETY: every pointer refers to a live host allocation whose length
    // matches exactly what the C side reads or writes (asserted above).
    let ms = unsafe {
        cuda_gemm_bench(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            nk as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

/// cuBLAS SGEMM on the same problem; an upper bound, not the mirror.
#[allow(clippy::too_many_arguments)]
pub fn gemm_cublas(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nk);
    assert_eq!(b.len(), nk * nj);
    assert_eq!(c.len(), ni * nj);
    let mut out = vec![0.0f32; ni * nj];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_gemm_cublas_bench(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            nk as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn twomm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nk);
    assert_eq!(b.len(), nk * nj);
    assert_eq!(c.len(), nj * nl);
    assert_eq!(d.len(), ni * nl);
    let mut out = vec![0.0f32; ni * nl];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_twomm_bench(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            nk as i32,
            nl as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn threemm(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    nm: usize,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nk);
    assert_eq!(b.len(), nk * nj);
    assert_eq!(c.len(), nj * nm);
    assert_eq!(d.len(), nm * nl);
    let mut out = vec![0.0f32; ni * nl];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_threemm_bench(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            d.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            nk as i32,
            nl as i32,
            nm as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn syrk(
    a: &[f32],
    c: &[f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), n * m);
    assert_eq!(c.len(), n * n);
    let mut out = vec![0.0f32; n * n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_syrk_bench(
            a.as_ptr(),
            c.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            m as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn syr2k(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), n * m);
    assert_eq!(b.len(), n * m);
    assert_eq!(c.len(), n * n);
    let mut out = vec![0.0f32; n * n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_syr2k_bench(
            a.as_ptr(),
            b.as_ptr(),
            c.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            m as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

pub fn atax(
    a: &[f32],
    x: &[f32],
    nx: usize,
    ny: usize,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), nx * ny);
    assert_eq!(x.len(), ny);
    let mut out = vec![0.0f32; ny];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_atax_bench(
            a.as_ptr(),
            x.as_ptr(),
            out.as_mut_ptr(),
            nx as i32,
            ny as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn bicg(
    a: &[f32],
    p: &[f32],
    r: &[f32],
    nx: usize,
    ny: usize,
    warmup: u32,
    iters: u32,
) -> Timed<(Vec<f32>, Vec<f32>)> {
    assert_eq!(a.len(), nx * ny);
    assert_eq!(p.len(), ny);
    assert_eq!(r.len(), nx);
    let mut s = vec![0.0f32; ny];
    let mut q = vec![0.0f32; nx];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_bicg_bench(
            a.as_ptr(),
            p.as_ptr(),
            r.as_ptr(),
            s.as_mut_ptr(),
            q.as_mut_ptr(),
            nx as i32,
            ny as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), (s, q))
}

#[allow(clippy::too_many_arguments)]
pub fn gesummv(
    a: &[f32],
    b: &[f32],
    x: &[f32],
    n: usize,
    alpha: f32,
    beta: f32,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);
    assert_eq!(x.len(), n);
    let mut out = vec![0.0f32; n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_gesummv_bench(
            a.as_ptr(),
            b.as_ptr(),
            x.as_ptr(),
            out.as_mut_ptr(),
            n as i32,
            alpha,
            beta,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

#[allow(clippy::too_many_arguments)]
pub fn mvt(
    a: &[f32],
    x1: &[f32],
    x2: &[f32],
    y1: &[f32],
    y2: &[f32],
    n: usize,
    warmup: u32,
    iters: u32,
) -> Timed<(Vec<f32>, Vec<f32>)> {
    assert_eq!(a.len(), n * n);
    assert!(x1.len() == n && x2.len() == n && y1.len() == n && y2.len() == n);
    let mut o1 = vec![0.0f32; n];
    let mut o2 = vec![0.0f32; n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_mvt_bench(
            a.as_ptr(),
            x1.as_ptr(),
            x2.as_ptr(),
            y1.as_ptr(),
            y2.as_ptr(),
            o1.as_mut_ptr(),
            o2.as_mut_ptr(),
            n as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), (o1, o2))
}

pub fn conv2d(a: &[f32], ni: usize, nj: usize, warmup: u32, iters: u32) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nj);
    let mut out = vec![0.0f32; ni * nj];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_conv2d_bench(
            a.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

pub fn conv3d(
    a: &[f32],
    ni: usize,
    nj: usize,
    nk: usize,
    warmup: u32,
    iters: u32,
) -> Timed<Vec<f32>> {
    assert_eq!(a.len(), ni * nj * nk);
    let mut out = vec![0.0f32; ni * nj * nk];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_conv3d_bench(
            a.as_ptr(),
            out.as_mut_ptr(),
            ni as i32,
            nj as i32,
            nk as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), out)
}

pub fn jacobi1d(
    a: &[f32],
    b: &[f32],
    n: usize,
    tsteps: usize,
    warmup: u32,
    iters: u32,
) -> Timed<(Vec<f32>, Vec<f32>)> {
    assert!(a.len() == n && b.len() == n);
    let mut oa = vec![0.0f32; n];
    let mut ob = vec![0.0f32; n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_jacobi1d_bench(
            a.as_ptr(),
            b.as_ptr(),
            oa.as_mut_ptr(),
            ob.as_mut_ptr(),
            n as i32,
            tsteps as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), (oa, ob))
}

pub fn jacobi2d(
    a: &[f32],
    b: &[f32],
    n: usize,
    tsteps: usize,
    warmup: u32,
    iters: u32,
) -> Timed<(Vec<f32>, Vec<f32>)> {
    assert!(a.len() == n * n && b.len() == n * n);
    let mut oa = vec![0.0f32; n * n];
    let mut ob = vec![0.0f32; n * n];
    // SAFETY: as above.
    let ms = unsafe {
        cuda_jacobi2d_bench(
            a.as_ptr(),
            b.as_ptr(),
            oa.as_mut_ptr(),
            ob.as_mut_ptr(),
            n as i32,
            tsteps as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), (oa, ob))
}

#[allow(clippy::too_many_arguments)]
pub fn fdtd2d(
    ex: &[f32],
    ey: &[f32],
    hz: &[f32],
    fict: &[f32],
    nx: usize,
    ny: usize,
    warmup: u32,
    iters: u32,
) -> Timed<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    assert!(ex.len() == nx * ny && ey.len() == nx * ny && hz.len() == nx * ny);
    let mut oex = vec![0.0f32; nx * ny];
    let mut oey = vec![0.0f32; nx * ny];
    let mut ohz = vec![0.0f32; nx * ny];
    // SAFETY: as above; `fict` is read for exactly `tmax = fict.len()` elements.
    let ms = unsafe {
        cuda_fdtd2d_bench(
            ex.as_ptr(),
            ey.as_ptr(),
            hz.as_ptr(),
            fict.as_ptr(),
            oex.as_mut_ptr(),
            oey.as_mut_ptr(),
            ohz.as_mut_ptr(),
            nx as i32,
            ny as i32,
            fict.len() as i32,
            warmup as i32,
            iters as i32,
        )
    };
    (us(ms), (oex, oey, ohz))
}
