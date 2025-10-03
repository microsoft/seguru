use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{PlaceContext, Visitor};
use rustc_middle::mir::{
    Body, Local, Location, Operand, Place, Rvalue, Statement, Terminator, TerminatorEdges,
    TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::fmt::DebugWithContext;
use rustc_mir_dataflow::{Analysis, Results, ResultsVisitor};
use tracing::{debug, error};

use crate::error::{GpuCodegenError, GpuCodegenResult};
use crate::scope::{ScopedFun, SyncScope};

type Domain = ThreadSyncDomain;

#[derive(Eq, PartialEq, Clone, Debug)]
struct AccessScopeState {
    // existing read/write accesses with scope info
    access: FxHashMap<Local, u64>,
    locs: FxHashMap<(Local, SyncScope), Vec<Location>>,
}

impl AccessScopeState {
    fn new(len: usize) -> Self {
        AccessScopeState { access: FxHashMap::default(), locs: FxHashMap::default() }
    }

    fn update_current(&mut self, local: Local, scope: SyncScope, loc: Location) {
        let entry = self.access.entry(local).or_insert(0);
        *entry |= scope.as_mask();
        self.locs.entry((local, scope)).or_default().push(loc);
    }

    fn clear(&mut self, scope: SyncScope) {
        for (local, sync_bits) in self.access.iter_mut() {
            *sync_bits &= !scope.as_mask(); // clear sync bits for this scope
        }
    }

    fn join(&mut self, other: &Self) -> bool {
        let mut changed = false;
        for (k, v) in other.access.iter() {
            let entry = self.access.entry(*k).or_insert(0);
            let before = *entry;
            *entry |= *v;
            if *entry != before {
                changed = true;
            }
        }
        for (k, loc) in other.locs.iter() {
            let entry = self.locs.entry(*k).or_default();
            let before = entry.len();
            for new_loc in loc.iter() {
                if !entry.contains(new_loc) {
                    entry.push(*new_loc);
                }
            }
            if entry.len() != before {
                changed = true;
            }
        }
        changed
    }

    // Find the conflicts between two AccessScopeStates
    fn conflicts(&self, other: &Self) -> FxHashSet<(Local, Location, Location)> {
        let mut conflicts = FxHashSet::default();
        for (local, sync_bits) in self.access.iter() {
            if let Some(other_bits) = other.access.get(local) {
                let conflict_bits = sync_bits & other_bits;
                if conflict_bits != 0 {
                    if conflict_bits == SyncScope::Block.as_mask() {
                        // Block scope sync is enough
                        if let Some(locs) = self.locs.get(&(*local, SyncScope::Block)) {
                            for loc in locs.iter() {
                                if let Some(other_locs) =
                                    other.locs.get(&(*local, SyncScope::Block))
                                {
                                    for other_loc in other_locs.iter() {
                                        conflicts.insert((*local, *loc, *other_loc));
                                    }
                                }
                            }
                        }
                    } else {
                        unimplemented!("multi-scope sync not yet supported");
                    }
                }
            }
        }
        conflicts
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct ThreadSyncDomainState {
    // existing read/write accesses with scope info
    pending: AccessScopeState,

    // current read/write accesses with scope info
    current: AccessScopeState,
}

impl ThreadSyncDomainState {
    fn new(len: usize) -> Self {
        ThreadSyncDomainState {
            pending: AccessScopeState::new(len),
            current: AccessScopeState::new(len),
        }
    }

    fn update_current(&mut self, local: Local, scope: SyncScope, loc: Location) {
        self.current.update_current(local, scope, loc);
    }

    fn clear_pending(&mut self, scope: SyncScope) {
        self.pending.clear(scope);
    }

    // join the state from the state in predecessors
    // Prev pending and current state are merged into the target state
    fn join(&mut self, prev: &Self) -> bool {
        let mut changed = false;
        changed |= self.pending.join(&prev.pending);
        changed |= self.pending.join(&prev.current);
        changed
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct ThreadSyncDomain {
    maybe_read: ThreadSyncDomainState,
    maybe_write: ThreadSyncDomainState,
    local_origin: FxHashMap<Local, DenseBitSet<Local>>, // Map from memory local to its original local
}

impl<'a> DebugWithContext<ThreadSyncAnalysis<'a, '_>> for ThreadSyncDomain {}

impl ThreadSyncDomain {
    fn empty(len: usize) -> Self {
        ThreadSyncDomain {
            maybe_read: ThreadSyncDomainState::new(len),
            maybe_write: ThreadSyncDomainState::new(len),
            local_origin: FxHashMap::default(),
        }
    }
}

impl rustc_mir_dataflow::JoinSemiLattice for ThreadSyncDomain {
    // join the state from the state in predecessors
    fn join(&mut self, prev: &Self) -> bool {
        let mut changes = false;
        for (i, v) in prev.local_origin.iter() {
            let entry =
                self.local_origin.entry(*i).or_insert(DenseBitSet::new_empty(v.domain_size()));
            let before = entry.domain_size();
            for bit in v.iter() {
                entry.insert(bit);
            }
            if entry.domain_size() != before {
                changes = true;
            }
        }

        changes |= self.maybe_read.join(&prev.maybe_read);
        changes |= self.maybe_write.join(&prev.maybe_write);
        changes
    }
}

/// The ThreadSyncAnalysis detects missing sync when
///     - write-read: write to thread-unique and then read to thread-shared mem.
///     - write-write: write to thread-unique and then rechunk/convert to atomic.
///     - read-write: read to thread-shared mem and then write to thread-unique.
/// The idea is to support multi-scope sync and track the sync scope bits.
/// Currently, we only support Block scope sync.
/// See sync_missing_xxx.rs tests
struct ThreadSyncAnalysis<'tcx, 'mir> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
}

impl<'tcx> Analysis<'tcx> for ThreadSyncAnalysis<'tcx, '_> {
    // Track the mutable argument locals and their aliases.
    type Domain = ThreadSyncDomain;
    type Direction = rustc_mir_dataflow::Forward;
    const NAME: &'static str = "ThreadSyncAnalysis";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        ThreadSyncDomain::empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {}

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        _location: Location,
    ) {
        let mut visitor = ThreadSyncMirVisitor {
            tcx: self.tcx,
            body: self.body,
            state,
            called_sync_threads: None,
        };
        visitor.visit_statement(stmt, _location);
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Self::Domain,
        terminator: &'mir Terminator<'tcx>,
        loc: Location,
    ) -> TerminatorEdges<'mir, 'tcx> {
        let mut visitor = ThreadSyncMirVisitor {
            tcx: self.tcx,
            body: self.body,
            state,
            called_sync_threads: None,
        };
        visitor.visit_terminator(terminator, loc);
        if let Some(sync_scope) = visitor.called_sync_threads {
            debug!("called_sync {:?}", loc);
            state.maybe_read.clear_pending(sync_scope);
            state.maybe_write.clear_pending(sync_scope);
        }
        terminator.edges()
    }
}

#[allow(dead_code)]
#[derive(Eq, PartialEq, Clone, Debug)]
struct LocalWithSyncScope {
    sync_scope: SyncScope,
    local: Local,
}

fn ty_mem_type_shared_or_global<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: &rustc_middle::ty::Ty<'tcx>,
) -> Option<bool> {
    if let Some(def) = ty.ty_adt_def() {
        if Some(def.did()) == tcx.get_diagnostic_item(rustc_span::Symbol::intern("gpu::GpuShared"))
        {
            return Some(true);
        } else if Some(def.did())
            == tcx.get_diagnostic_item(rustc_span::Symbol::intern("gpu::GpuGlobal"))
        {
            return Some(false);
        }
    }
    None
}

struct ThreadSyncMirVisitor<'a, 'tcx, 'mir> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'mir Body<'tcx>,
    called_sync_threads: Option<SyncScope>,
    state: &'a mut ThreadSyncDomain,
}

impl<'a, 'tcx, 'mir> std::ops::Deref for ThreadSyncMirVisitor<'a, 'tcx, 'mir> {
    type Target = ThreadSyncDomain;

    fn deref(&self) -> &Self::Target {
        self.state
    }
}

impl<'a, 'tcx, 'mir> std::ops::DerefMut for ThreadSyncMirVisitor<'a, 'tcx, 'mir> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state
    }
}

/// The analysis logic for sync_threads detection.
///
/// ## The analysis logic for sync_threads detection:
/// - Track the origin of a local.
/// - When a local is used for read/write, check if its origin and marked current state as read/write with corresponding scope.
/// - When the terminator is a call to sync function, clear the pending read/write states that correspond to the sync scope.
/// - At join point, merge the pending and current states from predecessors.
/// - Finally, we will get a state at each location with pending and current read/write states.
///
/// ## The result analysis logic for sync_threads detection:
///
/// Conflicts happens when
/// - a local is written and then read/written without sync in between
/// - a local is read and then written without sync in between
impl<'tcx> Visitor<'tcx> for ThreadSyncMirVisitor<'_, 'tcx, '_> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        let mutating_use = ctx.is_mutating_use();
        let local = place.local;
        let in_use = ctx.is_use();
        let state = &mut self.state;
        let origins = state
            .local_origin
            .entry(local)
            .or_insert(DenseBitSet::new_empty(self.body.local_decls.len()));
        if !in_use && origins.is_empty() {
            return;
        }
        // Find read/write after write to chunked variable.

        for origin in origins.iter() {
            let share_or_global: Option<bool> =
                ty_mem_type_shared_or_global(self.tcx, &self.body.local_decls[origin].ty);
            if share_or_global.is_none() {
                continue;
            };
            let scope = if share_or_global.unwrap() { SyncScope::Block } else { SyncScope::Grid };
            if mutating_use {
                state.maybe_write.update_current(origin, scope, loc);
            } else {
                state.maybe_read.update_current(origin, scope, loc);
            }
        }
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, loc: Location) {
        match rvalue {
            Rvalue::Ref(_, _, src) | Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                let origins = self
                    .state
                    .local_origin
                    .entry(src.local)
                    .or_insert(DenseBitSet::new_empty(self.body.local_decls.len()));
                if origins.is_empty() {
                    origins.insert(src.local);
                }
            }
            _ => {}
        };

        self.super_assign(place, rvalue, loc);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Visit the terminator to find reads/writes
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                let mut is_trusted_chunk_function = false;
                let mut sync_scope = None;
                if let Some(gpu_item) = &attr.gpu_item {
                    self.called_sync_threads =
                        ScopedFun::try_from(gpu_item.clone()).ok().and_then(|f| match f {
                            ScopedFun::Sync(scope) => Some(scope),
                            ScopedFun::NewChunk(scope) | ScopedFun::NewAtomic(scope) => {
                                is_trusted_chunk_function = true;
                                sync_scope = Some(scope);
                                None
                            }
                        });
                }
            }
        }
        self.super_terminator(terminator, location);
    }
}

pub(crate) struct ThreadSyncResultsVisitor {
    read_sync: FxHashSet<(Local, Location, Location)>,
    write_sync: FxHashSet<(Local, Location, Location)>,
}

impl ThreadSyncResultsVisitor {
    pub fn new<'tcx>(tcx: rustc_middle::ty::TyCtxt<'tcx>) -> Self {
        Self { read_sync: FxHashSet::default(), write_sync: FxHashSet::default() }
    }
}

impl<'mir, 'tcx> ResultsVisitor<'mir, 'tcx, ThreadSyncAnalysis<'tcx, 'mir>>
    for ThreadSyncResultsVisitor
{
    fn visit_after_primary_statement_effect(
        &mut self,
        _results: &mut Results<'tcx, ThreadSyncAnalysis<'tcx, 'mir>>,
        state: &Domain,
        stmt: &'mir Statement<'tcx>,
        location: Location,
    ) {
        // read-write conflicts
        let conflicts = state.maybe_read.pending.conflicts(&state.maybe_write.current);
        self.read_sync.extend(&conflicts);
        if !conflicts.is_empty() {
            error!("conflict read-write: {:?} at {:?}", stmt, conflicts);
        }

        // write-read or write-write conflicts
        let mut conflicts = state.maybe_write.pending.conflicts(&state.maybe_write.current);
        conflicts.extend(state.maybe_write.pending.conflicts(&state.maybe_read.current));
        self.write_sync.extend(&conflicts);
        if !conflicts.is_empty() {
            error!("conflict write-access: {:?} at {:?}", stmt, conflicts);
        }
    }
}

/// Analyze the body to check the mutable argument locals and their uses.
pub(crate) fn analyze_shared_access<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &Body<'tcx>,
    is_kernel_entry: bool,
) -> GpuCodegenResult<()> {
    let analysis = ThreadSyncAnalysis { tcx, body };
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = ThreadSyncResultsVisitor::new(tcx);
    results.visit_with(body, body.basic_blocks.iter_nodes(), &mut result_visitor);
    let mut missing_sync: bool = false;

    // Report missing sync error when needed.
    // If a shared mem origin is in NeedSync state and there is read after write.
    let missing_sync_err = |op: &str,
                            followed_op: &str,
                            sync_name: &str,
                            local: Local,
                            src_loc: Location,
                            dst_loc: Location| {
        let src_span = body.source_info(src_loc).span;
        let dst_span = body.source_info(dst_loc).span;
        let mut err = tcx.sess.dcx().struct_span_err(
            src_span,
            format!(
                "The {op} needs a `{sync_name}` called before other {followed_op}: {:?}",
                local
            ),
        );
        err.span_note(dst_span, format!("need `{sync_name}` before this {followed_op}"));
        err.emit();
    };
    for (local, src, dst) in result_visitor.write_sync.iter() {
        missing_sync_err("write", "read/write", "sync_threads", *local, *src, *dst);
        missing_sync = true;
    }
    for (local, src, dst) in result_visitor.read_sync.iter() {
        missing_sync_err("read", "write", "sync for write_scope", *local, *src, *dst);
        missing_sync = true;
    }
    if missing_sync {
        return Err(GpuCodegenError::MissingSyncThreads);
    }
    Ok(())
}
