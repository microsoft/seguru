use crate::dim::{DimType, DimTypeID, DimX, DimY, DimZ, block_dim, block_id, dim, thread_id};

trait PrivateTraitGuard {}

/// This trait is used to provide chunking scope information.
/// It is only implemented for global memory and shared memory.
#[expect(private_bounds)]
pub trait ChunkScope: PrivateTraitGuard + Clone {
    const TID_LEN: usize; // 6 for global mem, 3 for shared mem

    fn thread_ids() -> [usize; Self::TID_LEN];
    fn global_dim<D: DimType>() -> usize;
    fn global_id<D: DimType>(thread_ids: [usize; Self::TID_LEN]) -> usize;

    /// Provided methods.
    #[inline]
    #[gpu_codegen::device]
    fn global_id_x(thread_ids: [usize; Self::TID_LEN]) -> usize {
        Self::global_id::<DimX>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id_y(thread_ids: [usize; Self::TID_LEN]) -> usize {
        Self::global_id::<DimY>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id_z(thread_ids: [usize; Self::TID_LEN]) -> usize {
        Self::global_id::<DimZ>(thread_ids)
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_x() -> usize {
        Self::global_dim::<DimX>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_y() -> usize {
        Self::global_dim::<DimY>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim_z() -> usize {
        Self::global_dim::<DimZ>()
    }
}

#[derive(Copy, Clone)]
pub struct GlobalMemScope;
impl PrivateTraitGuard for GlobalMemScope {}
impl ChunkScope for GlobalMemScope {
    const TID_LEN: usize = 6;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [usize; Self::TID_LEN] {
        // global memory is accessible across blocks,
        // so we need block_id as well as thread_id.
        [
            thread_id::<DimX>(),
            thread_id::<DimY>(),
            thread_id::<DimZ>(),
            block_id::<DimX>(),
            block_id::<DimY>(),
            block_id::<DimZ>(),
        ]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [usize; Self::TID_LEN]) -> usize {
        thread_ids[D::DIM_ID + DimTypeID::Max as usize] * block_dim::<D>() + thread_ids[D::DIM_ID]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> usize {
        dim::<D>()
    }
}

#[derive(Copy, Clone)]
pub struct SharedMemScope;
impl PrivateTraitGuard for SharedMemScope {}
impl ChunkScope for SharedMemScope {
    const TID_LEN: usize = 3;

    #[inline]
    #[gpu_codegen::device]
    fn thread_ids() -> [usize; Self::TID_LEN] {
        // shared memory is shared within a block and so no block_id.
        [thread_id::<DimX>(), thread_id::<DimY>(), thread_id::<DimZ>()]
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_dim<D: DimType>() -> usize {
        block_dim::<D>()
    }

    #[inline]
    #[gpu_codegen::device]
    fn global_id<D: DimType>(thread_ids: [usize; Self::TID_LEN]) -> usize {
        thread_ids[D::DIM_ID]
    }
}
