#![feature(register_tool)]
#![register_tool(gpu_codegen)]
#![feature(stmt_expr_attributes)]
#![no_std]

extern crate gpu;

use gpu::prelude::*;

macro_rules! unroll4 {
    ($i:ident, $body:block) => {{
        {
            let $i: usize = 0;
            $body
        }
        {
            let $i: usize = 1;
            $body
        }
        {
            let $i: usize = 2;
            $body
        }
        {
            let $i: usize = 3;
            $body
        }
    }};
}

macro_rules! unroll8 {
    ($i:ident, $body:block) => {{
        {
            let $i: usize = 0;
            $body
        }
        {
            let $i: usize = 1;
            $body
        }
        {
            let $i: usize = 2;
            $body
        }
        {
            let $i: usize = 3;
            $body
        }
        {
            let $i: usize = 4;
            $body
        }
        {
            let $i: usize = 5;
            $body
        }
        {
            let $i: usize = 6;
            $body
        }
        {
            let $i: usize = 7;
            $body
        }
    }};
}

const BM: u32 = 128;
const BN: u32 = 128;
const BK: u32 = 8;
const TM: u32 = 8;
const TN: u32 = 8;
const BDIM_X: u32 = 16;

#[no_mangle]
#[gpu::kernel]
#[gpu::attr(nvvm_launch_bound(16, 16, 1, 2))]
pub fn gemm_add_relu_ptxas_kernel(
    x: &[Float4],
    w: &[Float4],
    bias: &[Float4],
    y: &mut [Float4],
    m: u32,
    n: u32,
    k: u32,
) {
    let _ = m;

    let tx = thread_id::<DimX>();
    let ty = thread_id::<DimY>();
    let bid_x = block_id::<DimX>();
    let bid_y = block_id::<DimY>();

    let bm = bid_y * BM;
    let bn = bid_x * BN;

    let tid = ty * BDIM_X + tx;
    let a_row = tid >> 1;
    let a_col = (tid & 1) << 2;

    let mut tile_a = GpuShared::<[f32; (BM * BK) as usize]>::zero();
    let mut tile_b = GpuShared::<[f32; (BN * BK) as usize]>::zero();

    let k4 = k >> 2;
    let a_col4 = a_col >> 2;

    // K-major shared layout: offset = (a_col + i0) * 128 + a_row.
    let load_map = reshape_map!([4] | [2, 8, 16] => layout: [t1, t2, i0, t0]);

    let mut acc = [[0.0f32; TN as usize]; TM as usize];

    let num_tiles = k / BK;
    let mut tstep: u32 = 0;
    while tstep < num_tiles {
        sync_threads();

        let k_base4 = tstep * (BK >> 2);

        {
            let mut ca = tile_a.chunk_mut(load_map);
            let idx_x = ((bm + a_row) * k4 + k_base4 + a_col4) as usize;
            let v: Float4 = x[idx_x];
            ca[0] = v[0];
            ca[1] = v[1];
            ca[2] = v[2];
            ca[3] = v[3];
        }
        {
            let mut cb = tile_b.chunk_mut(load_map);
            let idx_w = ((bn + a_row) * k4 + k_base4 + a_col4) as usize;
            let v: Float4 = w[idx_w];
            cb[0] = v[0];
            cb[1] = v[1];
            cb[2] = v[2];
            cb[3] = v[3];
        }

        sync_threads();

        let row_off = (ty * TM) as usize;
        let col_off = (tx * TN) as usize;

        unroll8!(kk, {
            let mut a_reg = [0.0f32; TM as usize];
            let mut b_reg = [0.0f32; TN as usize];

            unroll8!(ii, {
                a_reg[ii] = tile_a[kk * BM as usize + row_off + ii];
            });
            unroll8!(jj, {
                b_reg[jj] = tile_b[kk * BN as usize + col_off + jj];
            });

            unroll8!(ii, {
                let ai = a_reg[ii];
                unroll8!(jj, {
                    acc[ii][jj] += ai * b_reg[jj];
                });
            });
        });

        tstep += 1;
    }

    let bn4 = bn >> 2;
    let col4_base = bn4 + tx * 2;
    let b0: Float4 = bias[col4_base as usize];
    let b1: Float4 = bias[(col4_base + 1) as usize];

    let _ = n;
    let out_map = reshape_map!(
        [2, 8] | [16, grid_dim::<DimX>(), 16, grid_dim::<DimY>()]
        => layout: [i0, t0, t1, i1, t2, t3]
    );
    let mut y_tile = chunk_mut(y, out_map).open_tile();

    unroll8!(i, {
        let mut o0 = Float4::default();
        let mut o1 = Float4::default();
        unroll4!(j, {
            let mut v = acc[i][j] + b0[j];
            if v < 0.0 {
                v = 0.0;
            }
            o0[j] = v;
        });
        unroll4!(j, {
            let mut v = acc[i][j + 4] + b1[j];
            if v < 0.0 {
                v = 0.0;
            }
            o1[j] = v;
        });
        let mut row = y_tile.row_mut(i as u32);
        row[0u32] = o0;
        row[1u32] = o1;
    });
}
