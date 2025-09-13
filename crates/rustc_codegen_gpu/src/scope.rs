use crate::attr::GpuItem;

/// Synchronization scope levels in GPU execution.
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#id674
/// See thread_group::sync() in cuda/include/cooperative_groups.h
#[derive(Eq, PartialEq, PartialOrd, Copy, Clone, Debug, Hash)]
#[allow(dead_code)]
pub(crate) enum SyncScope {
    Warp,    // bar.warp.sync mask on .shared::{cta/cluster} (reserved)
    Block,   // bar.sync on .shared::cta
    Cluster, // barrier.cluster.arrive + barrier.cluster.wait on .shared::cluster (reserved)
    Grid,    // Reserved on .global
}

impl SyncScope {
    /// Returns the bitmask representing the scope and all nested scopes.
    /// Each scope includes all smaller (nested) scopes.  
    /// For example:
    /// When accessing a scope `A`, it also accesses all smaller scopes within `A`.
    /// For example, accessing `.grid` also accesses `.block`.
    ///
    /// When syncing in a scope `A`, it also synchronizes all smaller scopes within `A`.
    /// For example, syncing in `.grid` also syncs `.block`.
    pub(crate) const fn as_mask(self) -> u64 {
        match self {
            SyncScope::Warp => (1u64 << 32) - 1, // a warp has 32 threads and can be controlled via bar.warp.sync mask
            SyncScope::Block => (1u64 << 33) - 1,
            SyncScope::Cluster => (1u64 << 34) - 1,
            SyncScope::Grid => (1u64 << 35) - 1,
        }
    }
}

pub(crate) enum ScopedFun {
    NewChunk(SyncScope),
    Sync(SyncScope),
    NewAtomic(SyncScope),
}

impl TryFrom<GpuItem> for ScopedFun {
    type Error = ();

    fn try_from(item: GpuItem) -> Result<Self, Self::Error> {
        match item {
            GpuItem::SyncThreads => Ok(ScopedFun::Sync(SyncScope::Block)),
            GpuItem::DiagnoseOnly(name) => {
                if name == "gpu::chunk_mut" {
                    Ok(ScopedFun::NewChunk(SyncScope::Grid))
                } else if name == "gpu::shared_chunk_mut" {
                    Ok(ScopedFun::NewChunk(SyncScope::Block))
                } else if name == "gpu::sync::Atomic::new" {
                    Ok(ScopedFun::NewAtomic(SyncScope::Grid))
                } else {
                    Err(())
                }
            }
            _ => Err(()),
        }
    }
}
