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
    width: usize,
}

pub type MapLinear = MapLinearWithDim<3>;

impl<const N: usize> MapLinearWithDim<N> {
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(width: usize) -> Self {
        Self { width }
    }
}

/// # Safety
/// It is safe to use this mapping as long as the thread dimensions are properly
/// configured.
unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<1> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_y() == 1 && CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<2> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        ScopeUniqueMap::<CS>::map(&MapLinearWithDim::<3>::new(self.width), idx, thread_ids)
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for MapLinearWithDim<3> {
    type IndexType = usize;

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: usize, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
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
    pub x_size: usize,
}

impl Map2D {
    #[inline]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new(x_size: usize) -> Self {
        Self { x_size }
    }
}

unsafe impl<CS: ChunkScope> ScopeUniqueMap<CS> for Map2D {
    type IndexType = (usize, usize);

    #[inline]
    #[gpu_codegen::device]
    fn precondition(&self) -> bool {
        CS::global_dim_z() == 1
    }

    #[inline]
    #[gpu_codegen::device]
    fn map(&self, idx: Self::IndexType, thread_ids: [usize; TID_MAX_LEN]) -> (bool, usize) {
        let shape_x = self.x_size;
        let inner_x = idx.0;
        let inner_y = idx.1;
        let x = inner_x * CS::global_dim_x() + CS::global_id_x(thread_ids);
        let y = inner_y * CS::global_dim_y() + CS::global_id_y(thread_ids);
        (x < shape_x, shape_x * y + x)
    }
}

/// MapReshape should not be directly used by users.
/// Instead, use the macro `reshape_map!` to create a specialized version
/// that implements ScopeUniqueMap trait.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct MapReshape<const N: usize, const M: usize> {
    max_tid: usize,
    old_tid_w: [usize; N],
    new_tid_w: [isize; N],
    max_idx: usize,
    old_idx_w: [usize; M],
    new_idx_w: [isize; M],
    offset: isize,
}

/// User will not be able to use MapReshape as a regular mapping strategy directly.
impl<const N: usize, const M: usize, CS: ChunkScope> !ScopeUniqueMap<CS> for MapReshape<N, M> {}

impl<const N: usize, const M: usize> MapReshape<N, M> {
    #[doc(hidden)]
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new_with_weight_no_check(
        old_tid_w: [usize; N],
        new_tid_w: [isize; N],
        old_idx_w: [usize; M],
        new_idx_w: [isize; M],
        max_tid: usize,
        max_idx: usize,
        offset: isize,
    ) -> Self {
        Self { old_tid_w, new_tid_w, old_idx_w, new_idx_w, max_tid, max_idx, offset }
    }
    /// sizes: The shape of the input tensor, in terms of thread dimensions and index dimensions.
    /// sizes = [D_0, D_1, ..., D_{N-1}, D_N, ..., D_{N+M}]
    /// weights: The new strides for each dimension.
    /// index_places: The permutation of dimensions, also indicating the sorted order of weights
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new_with_weight<const LEN: usize>(
        sizes: [usize; LEN],
        weights: [isize; LEN],
        offset: isize,
    ) -> Self {
        // helper closure: product of slice elements after position `i`
        let suffix_prod = |arr: &[usize], i: usize| {
            let mut w = 1;
            arr[i..].iter().for_each(|v| w *= *v);
            w
        };
        for i in 0..LEN {
            let w = weights[i].unsigned_abs();
            let mut sum = 0;
            for j in 0..LEN {
                let w2 = weights[j].unsigned_abs();
                let s2 = sizes[j];
                if i == j {
                    continue;
                }
                if w2 <= w {
                    sum *= (s2 - 1) * w2;
                }
            }
            assert!(sum < w);
        }

        let mut old_tid_w = [0; N];
        let mut new_tid_w = [0; N];
        let mut old_idx_w = [0; M];
        let mut new_idx_w = [0; M];
        for i in 0..N {
            old_tid_w[i] = suffix_prod(&sizes[0..N], i + 1);
            new_tid_w[i] = weights[i];
        }
        for i in N..(N + M) {
            old_idx_w[i - N] = suffix_prod(&sizes, i + 1);
            new_idx_w[i - N] = weights[i];
        }
        Self {
            old_tid_w,
            new_tid_w,
            old_idx_w,
            new_idx_w,
            max_tid: suffix_prod(&sizes[0..N], 0),
            max_idx: suffix_prod(&sizes[N..N + M], 0),
            offset,
        }
    }

    /// sizes: The shape of the input tensor, in terms of thread dimensions and index dimensions.
    /// sizes = [D_0, D_1, ..., D_{N-1}, D_N, ..., D_{N+M}]
    /// index_places: The permutation of dimensions, indicating where each original dimension should go in the new layout.
    /// index_places p = [p_0, p_1, ..., p_{N+M-1}],
    /// Requires:
    ///   sizes[k] > 0,
    ///   index_places is a permutation of (0.. N+M)
    #[gpu_codegen::device]
    #[gpu_codegen::ret_sync_data(1000)]
    pub fn new<const LEN: usize>(sizes: [usize; LEN], index_places: [usize; LEN]) -> Self {
        let mut new_sizes = sizes;
        for i in 0..N + M {
            new_sizes[index_places[i]] = sizes[i];
        }
        // helper closure: product of slice elements after position `i`
        let suffix_prod = |arr: &[usize], i: usize| {
            let mut w = 1;
            arr[i..].iter().for_each(|v| w *= *v);
            w
        };

        let mut old_tid_w = [0; N];
        let mut new_tid_w = [0; N];
        let mut old_idx_w = [0; M];
        let mut new_idx_w = [0; M];
        for i in 0..N {
            old_tid_w[i] = suffix_prod(&sizes[0..N], i + 1);
            new_tid_w[i] = suffix_prod(&new_sizes, index_places[i] + 1) as isize;
        }
        for i in N..(N + M) {
            old_idx_w[i - N] = suffix_prod(&sizes, i + 1);
            new_idx_w[i - N] = suffix_prod(&new_sizes, index_places[i] + 1) as isize;
        }
        Self {
            old_tid_w,
            new_tid_w,
            old_idx_w,
            new_idx_w,
            max_tid: suffix_prod(&sizes[0..N], 0),
            max_idx: suffix_prod(&sizes[N..], 0),
            offset: 0,
        }
    }

    #[inline]
    #[gpu_codegen::device]
    pub fn map<CS: ChunkScope>(
        &self,
        extra_id: usize,
        thread_ids: [usize; TID_MAX_LEN],
    ) -> (bool, usize) {
        let tid = CS::global_id_x(thread_ids)
            + CS::global_dim_x()
                * (CS::global_id_y(thread_ids) + CS::global_dim_y() * CS::global_id_z(thread_ids));
        let mut remain = tid;
        let mut index = self.offset;
        for i in 0..N {
            index += (remain / self.old_tid_w[i]) as isize * self.new_tid_w[i];
            remain %= self.old_tid_w[i];
        }
        let valid = tid < self.max_tid && extra_id < self.max_idx;
        let mut remain = extra_id;
        for i in 0..M {
            let id = remain / self.old_idx_w[i];
            index += id as isize * self.new_idx_w[i];
            remain %= self.old_idx_w[i];
        }
        (valid, index as usize)
    }
}

#[macro_export]
macro_rules! count_literals {
    () => {0};
    ($head:literal $(, $tail:literal)*) => {1 + $crate::count_literals!($($tail),*)};
}

/// Maps a multi-dimensional thread/index layout into a reshaped linear index.
///
/// The `reshape_map!` macro generates a `MapReshape` struct that reshape
/// local thread ID and index to a shape and then create a linearized combination
/// of them according to a specified layout or weights to get global access index.
///
/// # Macro Signature
///
/// The macro takes either layout or weights to specify the new layout.
///
/// ## Using layout to get a permutation of thread and index to access the array.
/// ```text
/// reshape_map!(tsizes, extra_sizes => layout: index_places)
/// ```
/// - `tsizes`: array expression specifying the shape of the thread ID, including both thread:
///   `[D_0, D_1, ..., D_{N-1}]`. // go from high-order to low-order
/// - `extra_sizes`: array expression specifying the shape of the extra index.
///   `[D_{N+M-1}]`.
/// - `index_places`: permutation of dimensions in the new layout:
///   `[p_0, p_1, ..., p_{N+M-1}]` // go from high-order to low-order
///   - `0 <= p_k < N` for thread dimensions
///   - `N <= p_k < N+M` for index dimensions  
///   - `index_places` must be a valid and constant permutation.
///
/// # Behavior
/// - Translates `global_thread_id < D_0 * D_1 * ... * D_{N-1}` into multi-dimensional thread IDs:
///   `(tid_0, tid_1, ..., tid_{N-1})`.
/// - Merges thread IDs with index IDs:
///   `(tid_0, ..., tid_{N-1}, idx_0, ..., idx_{M-1})`.
/// - Accesses the array in permuted order:
///   `arr[id_{p_0}][id_{p_1}]...[id_{p_{N+M-1}}]`.
/// - Precomputes strides for efficient linear index computation in both old and new layouts.
///
/// # Safety
/// - Users cannot create `MapReshape` instances outside the macro without `unsafe {}`.
/// - Sizes must be non-zero, which is not checked at compile-time; `size = 0` triggers runtime errors and
///   should be treated as functionality errors and will not violate race-free guarantee.
/// - Guarantees safe mapping for valid permutations to ensure race-free chunking.
///
/// # Examples
///
/// See more tests in `chunk_scope::test_reshape_map`.
///
/// **Example 1:** no permutation
///
/// Similar to MapLinear(3) when num_thread = 4.
/// ```rust
/// gpu::reshape_map!([4] | [3] => layout: [0, 1]);
/// // thread shape: [4]
/// // extra index shape: [3]
/// // Access: arr[tid0][idx0]
/// // Linear mapping of tid: [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
/// ```
/// **Example 2:** permutation swaps a thread id and extra_id
///
/// Similar to MapLinear(1) when num_thread = 4, arr.len = 12.
/// ```rust
/// gpu::reshape_map!([4] | [4] => layout: [1, 0]);
/// // thread shape: [4]
/// // extra index shape: [3]
/// // Access: arr[idx0][tid0]
/// // Linear mapping of tid: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
/// ```
/// **Example 3:** software-defined thread dimension > 1
///
/// Allowing swap thread index, and so thread order changes.
/// ```rust
/// gpu::reshape_map!([2, 2] | [3] => layout: [1, 0, 2]);
/// // thread shape: [2, 2]
/// // extra index shape: [3]
/// // Access: arr[tid1][tid0][idx0]
/// // Linear mapping of tid: [0, 0, 0, 2, 2, 2, 1, 1, 1, 3, 3, 3]
/// ```
///
/// **Example 4:** swap tid and extra_id, with thread dimension > 1**
///
/// ```rust
/// gpu::reshape_map!([2, 2] | [3] => layout: [2, 0, 1]);
/// // thread shape: [2, 2]
/// // extra index shape: [3]
/// // Access: arr[tid1][idx0][tid0]
/// // Linear mapping of tid: [0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3]
/// ```
///
/// # Invalid Examples
///
/// **Example 1:** Invalid permutation (index out of range)
/// ```rust,compile_fail
/// gpu::reshape_map!([2] | [2, 3] => layout: [1, 2, 3]);
/// ```
/// **Example 2:** Invalid permutation (duplicate indices)
/// ```rust,compile_fail
/// gpu::reshape_map!([2] | [2, 3] => layout: [0, 0, 1]);
/// ```
/// **Example 3:** Invalid thread dimension (<1)
/// ```rust,compile_fail
/// gpu::reshape_map!([] | [2, 3] => layout: [0, 1, 2]);
/// ```
/// **Example 4:** Invalid index dimension (<1)
/// ```rust,compile_fail
/// gpu::reshape_map!([2, 2] | [] => layout: [0, 1]);
/// ```
///
/// ## Using `weights` to get a more flexible, compile-time unchecked, runtime checked mapping.
///
/// ```text
/// reshape_map!(tsizes, extra_sizes => weights: weights_array)
/// ```
/// - `tsizes`: array expression specifying the shape of the thread ID, including both thread
///   `[D_0, D_1, ..., D_{N-1}]`.
/// - `extra_sizes`: array expression specifying the shape of the extra index.
///   `[D_{N+M-1}]`.
/// - `weights_array`: array expression specifying the new weights for all dimensions.
///   `[W_0, W_1, ..., W_{N+M-1}]`.
///
/// # Behavior
/// - Translates `global_thread_id < D_0 * D_1 * ... * D_{N-1}` into multi-dimensional thread IDs:
///   `(tid_0, tid_1, ..., tid_{N-1})`.
/// - Merges thread IDs with index IDs:
///   `(tid_0, ..., tid_{N-1}, idx_0, ..., idx_{M-1})`.
/// - Accesses the array at
///   `tid_0 * W_0 + tid_1 * W_1 + ... + idx_0 * W_{N} + ... + idx_{M-1} * W_{N+M-1}`.
///
/// # Safety
///
/// - Users cannot create `MapReshape` instances outside the macro without `unsafe
/// {}`.
/// - Sizes and Weights guarantee unique indexing at runtime,
///   which is checked at runtime; invalid sizes or weights trigger runtime errors and
///   should be treated as functionality errors and will not violate race-free guarantee.
/// - If passing constant sizes and weights, the macro will check the validity at compile time.
///
/// # Examples
///
/// **Example 1:** Constant and correct weights.
/// ```rust
/// // Example 1: no permutation, similar to MapLinear(3) when num_thread = 4.
/// let map_reshape = gpu::reshape_map!([4] | [3] => weights: [3, 1]);
/// ```
///
/// **Example 2:** Constant but incorrect weights.
///
/// ```rust,compile_fail
/// // error: weight 1 too small for size 4
/// let map_reshape = gpu::reshape_map!([4] | [3] => weights: [1, 1]);
/// ```
#[macro_export]
macro_rules! reshape_map {
    ($($any: tt)*) => {
        gpu_macros::reshape_map!($crate, $($any)*)
    };
}
