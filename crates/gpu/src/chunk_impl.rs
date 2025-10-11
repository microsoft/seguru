use crate::chunk::ScopeUniqueMap;
use crate::chunk_scope::{ChunkScope, TID_MAX_LEN};
/// Linear mapping for 1D array.
/// N is the number of thread dimensions.
/// width is the chunking window.
/// The array is divided into chunks along threads until all elements are covered.
///
/// ## Examples
///
/// Given:
/// bdim = (x=4, y=1, z=1)
/// gdim = (x=1, y=1, z=1)
/// arr: [T; 8]
/// width = 2
///
/// With `width = 2`, the mapping assign two continuous elements to thread 0-3:
/// [0, 0, 1, 1, 2, 2, 3, 3]
///
/// width = 1, the mapping assign two non-continuous elements to thread 0-3:
/// [0, 1, 2, 3, 0, 1, 2, 3]
#[derive(Copy, Clone)]
pub struct MapLinearWithDim<const N: usize = 3> {
    width: u32,
}

pub type MapLinear = MapLinearWithDim<3>;

impl<const N: usize> MapLinearWithDim<N> {
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(width: u32) -> Self {
        Self { width }
    }
}

/// # Safety
/// It is safe to use this mapping as long as the thread dimensions are properly
/// configured.
unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<1> {
    type IndexType = u32;
    type GlobalIndexType = u32;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_y() == 1 && CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(
        &self,
        idx: Self::IndexType,
        thread_ids: [u32; TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<2> {
    type IndexType = u32;
    type GlobalIndexType = u32;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(
        &self,
        idx: Self::IndexType,
        thread_ids: [u32; TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<3> {
    type IndexType = u32;
    type GlobalIndexType = u32;

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: u32, thread_ids: [u32; TID_MAX_LEN]) -> (bool, Self::GlobalIndexType) {
        let x_id = CS::global_id_x(thread_ids);
        let y_id = CS::global_id_y(thread_ids);
        let z_id = CS::global_id_z(thread_ids);
        let global_thread_id = x_id + (z_id * CS::global_dim_y() + y_id) * CS::global_dim_x();
        let stride = self.width;
        let total_dim = CS::global_dim_x() * CS::global_dim_y() * CS::global_dim_z();
        (true, idx % stride + (idx / stride) * stride * total_dim + global_thread_id * stride)
    }
}

/// This mapping strategy is useful when we want to reshape a 1D array into a 2D
/// array and then distribute one element to a thread one by one until consuming
/// all. It creates a new non-continuous partition for each thread.
/// - IndexType is (usize, usize)
///
/// Example:
/// Example:
/// - array: [T; 20]
/// - x_size = 5 => y_size = 4
/// - dim: x=2, y=2, z=1
/// ```text
/// 0   1   2   3   4   5
/// ┌───┬───┬───┬───┬───┐
/// │0,0│1,0│0,0│1,0│0,0│
/// ├───┼───┼───┼───┼───┤
/// │0,1│1,1│0,1│1,1│0,1│
/// ├───┼───┼───┼───┼───┤
/// │0,0│1,0│0,0│1,0│0,0│
/// ├───┼───┼───┼───┼───┤
/// │0,1│1,1│0,1│1,1│0,1│
/// └───┴───┴───┴───┴───┘
/// ```
/// In this case,
/// - thread(1,0) and (1,1) should only have access to a shape of 2*2 = 4 elements.
/// - thread (0,0) and (0,1) have access to a shape of 3*2 = 6 elements.
#[derive(Copy, Clone)]
pub struct Map2D {
    pub x_size: u32,
}

impl Map2D {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(x_size: u32) -> Self {
        Self { x_size }
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for Map2D {
    type IndexType = (u32, u32);
    type GlobalIndexType = u32;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(
        &self,
        idx: Self::IndexType,
        thread_ids: [u32; TID_MAX_LEN],
    ) -> (bool, Self::GlobalIndexType) {
        let shape_x = self.x_size;
        let inner_x = idx.0;
        let inner_y = idx.1;
        let x = inner_x * CS::global_dim_x() + CS::global_id_x(thread_ids);
        let y = inner_y * CS::global_dim_y() + CS::global_id_y(thread_ids);
        (x < shape_x, shape_x * y + x)
    }
}

/// Maps a multi-dimensional local index and thread layout into a reshaped global index.
/// The idea is that the user need to specify the dimensions of logical index (local + thread ids), and the target tensor dimensions, then we map the logical index into the corresponding target multi-dimension tensor.
/// We have:
/// - Logical index —  e.g., (threadId, local_access_id) in CUDA.
/// - Target tensor dimensions — e.g., a tensor of shape $(\hat{D}_0, \hat{D}_1, ... \hat{D}_{N+M})$
///
/// We  want to map the logical linear or multi-dimensional index into the corresponding tensor element, possibly in a flattened memory layout.
/// To ensure we can access the tensor properly, developers need to do linear2vec operation to convert threadid to $tids[D_{N+M}][...][D_N]$, and local_access_id to $lids[D_N][...][D_0]$
/// The `reshape_map!` macro generates a `MapReshape` struct that reshapes
/// local thread ID and index to a shape and then creates a linearized combination
/// of them according to a specified layout or weights to get a global access index.
/// ## Macro Signature
/// The macro takes dimension sizes and optional `layout` `offset` to specify the new layout.
/// ```text
/// gpu::reshape_map!(
///     [($local_id_dim, $tensor_dim?),*]  // local id + corresponding tensor dimension
///     |
///     [($thread_id_dim, $tensor_dim?),*] // thread id + corresponding tensor dimension
///     =>
///     layout: [-?i1, -?i2, ...]          // permutation of dimensions in the new layout
///     offset: $offset                // optional offset (default 0)
/// )
/// ```
/// - `lid_tensor_Dims`: array expression specifying the shape of local index, and the array shape corresponding to the local index:  
///   ```math
///   [(D_0, TD_0), (D_1, TD_1), ..., (D_{N-1}, TD_{N-1})]
///   ```
/// - `tid_tensor_Dims`: array expression specifying the shape of thread index, and the array shape corresponding to the local index:  
///   $[(D_N, TD_N), ..., (D_{N+M-1}, TD_{N+M-1)]$
/// - When $TD_k$ is omitted, it is assumed to be $D_k$.
/// - `permutation`: permutation of dimensions in the new layout:  
///   $[p_0, p_1, ..., p_{N+M-1}]$ (low → high dimension)
///   - $0 \le p_k < N$ for thread dimensions  
///   - $N \le p_k < N+M$ for index dimensions  
///   - Must be a valid constant permutation  [0, 1, 2, 3] or [i0, i1, t0, t1]
///   - If omitted, defaults to $[0, 1, ..., N+M-1]$
///   - Negative index `-k` is allowed to indicate $TD_k - 1 - id_k$ instead of $id_k$
///   - `-0` is allowed  
///   - `p_k` can be a constant literal or a readable name like `t<k-N>` or `i<k>` for thread and index dimensions
/// ## Behavior
/// - Translates `linear_thread_id` and `local_id` to software-defined multi-dimensional thread IDs:  
///   $lid_k = lid / \prod_{j=0}^{k - 1}(D_j) \mod D_k$
///   $lid = \sum_{k=0}^{N}(lid_k * \prod_{j=0}^{k - 1}(D_j))$
///   $tid_k = tid / \prod_{j=N}^{k - 1}(D_{j+N}) \mod D_{k+N}$
///   $tid = (tid_0, tid_1, ..., tid_{N-1})$ (low → high)
/// - Merges thread IDs with index IDs:  
///   $id = (lid_0, ..., lid_{M-1}, tid_0, ..., tid_{N-1})$ (low → high)
///   Thus, the logical index has dimention $idxDim = [D_0, D_1, ... D_{N+M+1}]$
/// - Treats array as shape:  
///   $\hat{D} = [TD_0, TD_1, .., TD_{N+M-1]$
/// - If $TD_k \ne $D_k$, some threads or indices will be skipped.  
///   Valid access range:  
///   $0 \le id_k < \min(D_k, TD_k)$
/// - Apply reversed flag to index
///   id'_k =if reversed_k {TD_k - 1 - id_k} else {id_k}
/// - Global ID
///   $globalid =  sum_{k=0}^{N+M}( id'_{p_{k}} prod_{j=0}^{k - 1}(TD_{p_j}))$
/// - Accesses the array in permuted order
///   $arr[id'_{p_0}][id'_{p_1}]...[id'_{p_{N+M-1}}]$
///
/// ## Safety
/// - Users cannot create `MapReshape` instances outside the macro without `unsafe {}`.
/// - Sizes must be non-zero; `size = 0` triggers runtime errors and should be treated as functionality errors.  
///   This will not violate race-free guarantees.
/// - Guarantees safe mapping for valid permutations to ensure race-free chunking.
/// ## Examples
/// See more tests in `chunk_scope::test_reshape_map`.
/// ### Example 1: No permutation
/// Similar to `MapLinear(3)` when `num_thread = 4`.
/// ```rust
/// gpu::reshape_map!([3] | [4]  => layout: [0, 1]);
/// ```
/// which is equivalent to
/// ```rust
/// gpu::reshape_map!([3] | [4] => layout: [i0, t0]);
/// // local index shape: [3]
/// // thread shape: [4]
/// // Access: arr[tid0][idx0]
/// // access -> tid: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
/// ```
/// ### Example 2: Permutation swaps a thread id and local_id
/// Similar to `MapLinear(1)` when `num_thread = 4, arr.len = 12`.
/// ```rust
/// gpu::reshape_map!([3] | [4] => layout: [1, 0]);
/// ```
/// which is equivalent to
/// ```rust
/// gpu::reshape_map!([3] | [4] => layout: [t0, i0]);
/// // local index shape: [3]
/// // thread shape: [4]
/// // array shape: arr[4][3]
/// // Access: arr[idx0][tid0]
/// // access -> tid: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
/// ```
/// ### Example 3: Software-defined thread dimension > 1
/// ```rust
/// gpu::reshape_map!([3] | [2, 2] => layout: [0, 2, 1]);
/// ```
/// which is equivalent to
///
///  ```rust
/// gpu::reshape_map!([3] | [2, 2] => layout: [i0, t1, t0]);
/// // local index shape: [3]
/// // thread shape: [2, 2]
/// // Access: arr[tid0][tid1][idx0]
/// // access -> tid: [0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3]
/// ```
/// ### Example 4: Swap tid and extra_id, thread dimension > 1
/// ```rust
/// gpu::reshape_map!([3] | [2, 2] => layout: [2, 0, 1]);
/// gpu::reshape_map!([3] | [2, 2] => layout: [t1, i0, t0]);
/// // local index shape: [3]
/// // thread shape: [2, 2]
/// // array shape: arr[2][2][3]
/// // Access: arr[tid0][idx0][tid1]
/// // access -> tid: [0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3]
/// ```
/// ### Example 5: Reverse a thread dimension
/// ```rust
/// gpu::reshape_map!([3] | [2, 2] => layout: [0, -1, 2]);
/// gpu::reshape_map!([3] | [2, 2] => layout: [i0, -t0, t1]);
/// // local index shape: [3]
/// // thread shape: [2, 2]
/// // Access: arr[tid1][(max_tid0 - tid0)][idx0]
/// ```
/// ### Example 6: Skip some threads by setting a smaller new size
/// ```rust
/// gpu::reshape_map!([3] | [2, (2, 1)] => layout: [0, 1, 2]);
/// gpu::reshape_map!([3] | [2, (2, 1)] => layout: [i0, t0, t1]);
/// // local index shape: [3]
/// // thread shape: [2, 2]
/// // array shape: arr[1][2][3]
/// // Access: arr[tid1][tid0][idx0]
/// // valid access range:
/// // 0 <= lid0 < 3
/// // 0 <= tid0 < 2
/// // 0 <= tid1 < 1
/// // access -> tid: [0, 0, 0, 1, 1, 1, _, _, _, _, _, _]
/// ```
/// ### Example 7: Skip some data by setting a larger size
/// ```rust
/// gpu::reshape_map!([(3, 4)] | [2, 2] => layout: [0, 1, 2]);
/// gpu::reshape_map!([(3, 4)] | [2, 2] => layout: [i0, t0, t1]);
/// // local index shape: [3]
/// // thread shape: [2, 2]
/// // array shape: arr[2][2][4]
/// // Access: arr[tid1][tid0][idx0]
/// // access -> tid: [0, 0, 0, _, 1, 1, 1, _, 2, 2, 2, _, 3, 3, 3, _, ...]
/// ```
/// ## Invalid Examples
/// ### Example 1: Invalid permutation (index out of range)
/// ```rust,compile_fail
/// gpu::reshape_map!([2] | [2, 3] => layout: [1, 2, 3]);
/// ```
/// ### Example 2: Invalid permutation (duplicate indices)
/// ```rust,compile_fail
/// gpu::reshape_map!([2] | [2, 3] => layout: [0, 0, 1]);
/// gpu::reshape_map!([2] | [2, 3] => layout: [t0, t0, i0]);
/// ```
/// ### Example 3: Invalid thread dimension (<1)
/// ```rust,compile_fail
/// gpu::reshape_map!([2, 3] | [] => layout: [0, 1, 2]);
/// ```
/// ### Example 4: Invalid index dimension (<1)
/// ```rust,compile_fail
/// gpu::reshape_map!( [] | [2, 2] => layout: [0, 1]);
/// ```
#[macro_export]
macro_rules! reshape_map {
    ($($any: tt)*) => {
        gpu_macros::reshape_map!($crate, $($any)*)
    };
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::chunk_scope::test::MockWarp2ThreadScope;

    macro_rules! assert_access_map {
        ($cs:ty, $m:expr, $thread_num:expr, $id_num:expr, $expected:expr) => {
            let access = get_access_map::<$thread_num, $cs>(&$m, $id_num);
            assert!(access == $expected, "access_map = {:?}, expected = {:?}", access, $expected)
        };
    }

    fn get_access_map<const NTHREADS: usize, CS>(
        map: &impl ScopeUniqueMap<CS, IndexType = u32>,
        n: usize,
    ) -> alloc::vec::Vec<isize>
    where
        CS: ChunkScope<FromScope = crate::cg::ThreadWarpTile<NTHREADS>, ToScope = crate::cg::Thread>,
    {
        let mut access_map = alloc::vec![-1isize; n];
        assert!(NTHREADS <= 32);
        for t in 0..NTHREADS {
            for i in 0..n {
                let tids = [0, t as u32, 0, 0, 0, 0];
                let (valid, mapped_idx) = ScopeUniqueMap::<CS>::map(map, i as u32, tids);
                let mapped_idx: usize = mapped_idx.into() as usize;
                if valid && mapped_idx < n {
                    access_map[mapped_idx] = t as _;
                }
            }
        }
        access_map
    }

    #[test]
    pub(crate) fn test_reshape_map_example3() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        let map_reshape = reshape_map!([3] | [2, 2] => layout: [0, 2, 1]);
        assert_access_map!(S, map_reshape, 4, 12, [0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3]);
    }

    #[test]
    pub(crate) fn test_reshape_map_example4() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        let map_reshape = reshape_map!([3] | [2, 2] => layout: [2, 0, 1]);
        assert_access_map!(S, map_reshape, 4, 12, [0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3]);
        let map_reshape = reshape_map!([3] | [2, 2] => layout: [t1, i0, t0]);
        assert_access_map!(S, map_reshape, 4, 12, [0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3]);
    }

    #[test]
    pub(crate) fn test_reshape_map_example5() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        let map_reshape = crate::reshape_map!([3] | [2, 2] => layout: [i0, -t0, t1]);
        assert_access_map!(S, map_reshape, 4, 12, [1, 1, 1, 0, 0, 0, 3, 3, 3, 2, 2, 2]);
    }

    #[test]
    pub(crate) fn test_reshape_map_example5_2() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        let map_reshape = crate::reshape_map!([3] | [2, 2] => layout: [-t0, -i0, t1]);
        assert_access_map!(S, map_reshape, 4, 12, [1, 0, 1, 0, 1, 0, 3, 2, 3, 2, 3, 2]);
    }

    #[test]
    pub(crate) fn test_reshape_map_example6() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        // Skip some threads by setting a smaller new size
        let map_reshape = crate::reshape_map!([3] | [2, (2, 1)] => layout: [i0, t0, t1]);
        // warp2thread and so use tid[1] only
        assert_access_map!(S, map_reshape, 4, 12, [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1]);
    }

    #[test]
    pub(crate) fn test_reshape_map_example7() {
        type S = MockWarp2ThreadScope<4, 0>; // a group of 4 threads.
        //Skip some data by setting a larger size
        let map_reshape = crate::reshape_map!([(3, 4)] | [2, 2] => layout: [i0, t0, t1]);
        assert_access_map!(S, map_reshape, 4, 12, [0, 0, 0, -1, 1, 1, 1, -1, 2, 2, 2, -1]);
    }
}
