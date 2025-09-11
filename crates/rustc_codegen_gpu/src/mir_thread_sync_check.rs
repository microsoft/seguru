use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    Body, Local, Location, Operand, Place, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::Ref;
use rustc_mir_dataflow::{Analysis, Results, ResultsVisitor};
use tracing::debug;

use crate::attr::GpuItem;
use crate::error::{GpuCodegenError, GpuCodegenResult};

// If the function is a kernel entry, we find out all mutable args.
// Since kernel entry should use GlobalMem for mutable args, we check if
// mutable args are used for mutation.
struct ThreadSyncAnalysis {
    is_kernel_entry: bool,
}

type Domain = DenseBitSet<Local>;

/// Find all mutable argument locals and propagate their aliases through the forward analysis.
impl<'tcx> Analysis<'tcx> for ThreadSyncAnalysis {
    // Track the mutable argument locals and their aliases.
    type Domain = DenseBitSet<Local>;
    type Direction = rustc_mir_dataflow::Forward;
    const NAME: &'static str = "ThreadSyncAnalysis";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // Add initial mutable argument locals
        if !self.is_kernel_entry {
            return;
        }
        for local in body.args_iter() {
            let local_decl = &body.local_decls[local];
            if local_decl.mutability.is_mut() {
                state.insert(local);
            }
            if let Ref(_, _, rustc_middle::ty::Mutability::Mut) = local_decl.ty.kind() {
                state.insert(local);
            }
        }
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Self::Domain,
        stmt: &Statement<'tcx>,
        _location: Location,
    ) {
        if !self.is_kernel_entry {
            return;
        }
        // Propagate aliases through moves/copies
        if let StatementKind::Assign(box (lhs, Rvalue::Ref(_, _, src))) = &stmt.kind {
            if state.contains(src.local) {
                state.insert(lhs.local);
            }
        }
        if let StatementKind::Assign(box (
            lhs,
            Rvalue::Use(Operand::Copy(src) | Operand::Move(src)),
        )) = &stmt.kind
        {
            if state.contains(src.local) {
                state.insert(lhs.local);
            }
        }
    }
}

/// https://docs.nvidia.com/cuda/parallel-thread-execution/#id674
/// .cta = block scope
/// The set of all threads executing in the same CTA as the current thread.
/// .cluster
/// The set of all threads executing in the same cluster as the current thread.
/// .gpu
/// The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device.
/// .sys
/// The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself.
#[derive(Debug, Clone, Copy)]
enum SyncScope {
    Block,
    Cluster,
}

enum ScopedFun {
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
                    Ok(ScopedFun::NewChunk(SyncScope::Cluster))
                } else if name == "gpu::shared_chunk_mut" {
                    Ok(ScopedFun::NewChunk(SyncScope::Block))
                } else if name == "gpu::sync::Atomic::new" {
                    Ok(ScopedFun::NewAtomic(SyncScope::Cluster))
                } else {
                    Err(())
                }
            }
            _ => Err(()),
        }
    }
}

#[allow(dead_code)]
struct LocalWithSyncScope {
    sync_scope: SyncScope,
    local: Local,
}

#[derive(Debug)]
enum SharedState {
    // .0: scope that needs sync
    // .1: The location of the write to the chunk, the reason why we need sync
    // .2: The locations of the reads/writes to shared mem after the write to the chunk
    NeedSync(SyncScope, Location, Vec<Location>),
    Synced,
}

// ThreadSyncDataFlowVistors detects two issues:
// 1) Mutable argument is used in non-chunking/atomic functions.
// 2) Missing sync_threads between a write to a shared chunk and other read/write to shared memory.
pub(crate) struct ThreadSyncDataFlowVistors<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    writes: Vec<(Place<'tcx>, Location)>, // This is mostly not needed after represent &mut as GlobalMem, but keep it for now as a defense in depth.
    mem_state: FxHashMap<Local, SharedState>, // mem to its state
    chunk_map: FxHashMap<Local, LocalWithSyncScope>, // Map from chunk local to global/shared local
    local_origin: FxHashMap<Local, Vec<Local>>, // Map from memory local to its original local
}

impl<'tcx> ThreadSyncDataFlowVistors<'tcx> {
    pub fn new(
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    ) -> Self {
        Self {
            tcx,
            writes: Vec::new(),
            local_decls,
            chunk_map: FxHashMap::default(),
            mem_state: FxHashMap::default(),
            local_origin: FxHashMap::default(),
        }
    }
}

/// Visitor to collect reads/writes of mutable arguments.
/// It skips the chunk function to allow the trusted GPU memory partition.
struct ThreadSyncMirVisitor<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    state: &'a Domain,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    called_sync_threads: Option<SyncScope>,
    inside_fcall: bool,
    flow_visitor: &'a mut ThreadSyncDataFlowVistors<'tcx>,
}

impl<'a, 'tcx> std::ops::Deref for ThreadSyncMirVisitor<'a, 'tcx> {
    type Target = ThreadSyncDataFlowVistors<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.flow_visitor
    }
}

impl<'a, 'tcx> std::ops::DerefMut for ThreadSyncMirVisitor<'a, 'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.flow_visitor
    }
}

impl<'a, 'tcx> ThreadSyncMirVisitor<'a, 'tcx> {
    fn is_mut_arg(&self, local: Local) -> bool {
        self.local_decls[local].mutability.is_mut() && !self.is_mut_ref_arg(local)
    }

    fn is_mut_ref_arg(&self, local: Local) -> bool {
        matches!(self.local_decls[local].ty.kind(), Ref(_, _, rustc_middle::ty::Mutability::Mut))
    }
}

/// The analysis logic for mutable argument access and sync_threads detection.
///
/// ## The analysis logic for mutable argument access (global mem):
/// - Track the mutable argument locals and their aliases.
/// - If it is used for mutable argument to a trusted chunk function, it is allowed.
/// - If it is used for mutation in other cases, record the location and report error.
/// ## The analysis logic for sync_threads detection (shared mem):
/// - Track the origin of a local.
/// - Track the creation of shared chunk and mapping it to its shared mem origin.
/// - When the shared chunk is mutated, marked the shared mem origin as NeedSync, the current loc as mut_loc.
/// - If a place is used and its origin is in NeedSync state, it means
///   we need a sync_thread between the mut_loc and the current loc.
impl<'tcx> Visitor<'tcx> for ThreadSyncMirVisitor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext};
        let flow_visitor = &mut self.flow_visitor;
        let origins = if let Some(origins) = flow_visitor.local_origin.get(&place.local) {
            origins
        } else {
            &vec![place.local]
        };

        for origin in origins {
            if let Some(SharedState::NeedSync(_, mut_loc, unsync_locs)) =
                flow_visitor.mem_state.get_mut(origin)
            {
                unsync_locs.push(loc);
            }
        }

        if self.state.contains(place.local) {
            match ctx {
                PlaceContext::MutatingUse(
                    MutatingUseContext::Borrow | MutatingUseContext::RawBorrow,
                ) => {
                    // Allow let a = &mut b since a is not used to mutate b yet.
                }
                PlaceContext::MutatingUse(_) => {
                    // If projection is empty, it is assign value (e.g., let x =
                    // &b[i];) to the local instead of dereferencing it (e.g.,
                    // *x = 0).
                    if !place.projection.is_empty() || self.is_mut_arg(place.local) {
                        debug!("Disallow mutate mutable argument: {:?} used as {:?}", place, ctx);
                        self.writes.push((*place, loc));
                    }
                }
                PlaceContext::NonMutatingUse(
                    NonMutatingUseContext::RawBorrow
                    | NonMutatingUseContext::Inspect
                    | NonMutatingUseContext::FakeBorrow
                    | NonMutatingUseContext::SharedBorrow,
                ) => {
                    // Allow let a = b; or let a = &b; let len = b.len();
                }
                PlaceContext::NonMutatingUse(_) => {
                    if !place.projection.is_empty()
                        || self.inside_fcall
                        || self.is_mut_arg(place.local)
                    {
                        debug!(
                            "Disallow use mutable argument via index/function: {:?} used as {:?}",
                            place, ctx
                        );
                        self.writes.push((*place, loc));
                    }
                }
                PlaceContext::NonUse(_) => {}
            }
        }
        // If the place is a chunked variable, update its shared memory state.
        // If the operation is a write, mark it as NeedSync.
        if self.chunk_map.contains_key(&place.local) {
            if let PlaceContext::MutatingUse(_) = ctx {
                let flow_visitor = &mut self.flow_visitor;
                let chunk_from = &flow_visitor.chunk_map[&place.local];
                let origins =
                    if let Some(origins) = flow_visitor.local_origin.get(&chunk_from.local) {
                        origins
                    } else {
                        &vec![chunk_from.local]
                    };
                for origin in origins {
                    debug!("Marking var {:?} as NeedSync: {:?}, {:?}", origin, place.local, loc);
                    let sync_scope = chunk_from.sync_scope;
                    flow_visitor
                        .mem_state
                        .insert(*origin, SharedState::NeedSync(sync_scope, loc, vec![]));
                }
            }
        }
    }

    fn visit_assign(&mut self, place: &Place<'tcx>, rvalue: &Rvalue<'tcx>, loc: Location) {
        match rvalue {
            Rvalue::Ref(_, _, src) | Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                let origin = if let Some(origin) = self.local_origin.get(&src.local) {
                    origin.clone()
                } else {
                    vec![src.local]
                };
                if self.local_origin.get_mut(&place.local).is_none() {
                    self.local_origin.insert(place.local, origin);
                } else {
                    self.local_origin.get_mut(&place.local).unwrap().extend_from_slice(&origin);
                }
            }
            _ => {}
        };

        self.super_assign(place, rvalue, loc);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Visit the terminator to find reads/writes
        self.inside_fcall = true;
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

                // Skip the chunk function to allow the trusted GPU memory partition.
                // TODO: checks on !is_trusted_chunk_function is not needed in the future if we represent &'a mut T as GlobalMem<'a, T>, defense in depth for now.
                if is_trusted_chunk_function {
                    self.visit_span(*fn_span);
                    self.visit_operand(func, location);
                    // Only skip the first argument.
                    for arg in args.iter().skip(1) {
                        self.visit_operand(&arg.node, location);
                    }
                    self.visit_place(
                        destination,
                        PlaceContext::MutatingUse(MutatingUseContext::Call),
                        location,
                    );
                    let global_or_shared_var = args[0].node.place().unwrap().local;
                    let chunked_var = destination.local;
                    self.flow_visitor.chunk_map.insert(
                        chunked_var,
                        LocalWithSyncScope {
                            sync_scope: sync_scope.unwrap(),
                            local: global_or_shared_var,
                        },
                    );
                    self.inside_fcall = false;
                    return;
                }
            }
        }
        self.super_terminator(terminator, location);
        self.inside_fcall = false;
    }
}

impl<'mir, 'tcx> ResultsVisitor<'mir, 'tcx, ThreadSyncAnalysis>
    for ThreadSyncDataFlowVistors<'tcx>
{
    fn visit_after_primary_statement_effect(
        &mut self,
        _results: &mut Results<'tcx, ThreadSyncAnalysis>,
        state: &Domain,
        stmt: &'mir Statement<'tcx>,
        location: Location,
    ) {
        ThreadSyncMirVisitor {
            tcx: self.tcx,
            local_decls: self.local_decls,
            state,
            flow_visitor: self,
            called_sync_threads: None,
            inside_fcall: false,
        }
        .visit_statement(stmt, location);
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        _results: &mut Results<'tcx, ThreadSyncAnalysis>,
        state: &Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) {
        let mut visitor = ThreadSyncMirVisitor {
            tcx: self.tcx,
            local_decls: self.local_decls,
            state,
            flow_visitor: self,
            called_sync_threads: None,
            inside_fcall: false,
        };
        visitor.visit_terminator(terminator, location);

        // If sync_threads is called, all chunk states become Synced.
        if let Some(sync_scope) = visitor.called_sync_threads {
            for (local, state) in self.mem_state.iter_mut() {
                debug!("Marking shared var {:?} as synced: {:?}", local, location);
                if let SharedState::NeedSync(sync_scope, _, _) = state {
                    *state = SharedState::Synced;
                }
            }
        }
    }
}

/// Analyze the body to check the mutable argument locals and their uses.
pub(crate) fn analyze_shared_access<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'tcx Body<'tcx>,
    is_kernel_entry: bool,
) -> GpuCodegenResult<()> {
    let analysis = ThreadSyncAnalysis { is_kernel_entry };
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = ThreadSyncDataFlowVistors::new(tcx, &body.local_decls);
    results.visit_with(body, body.basic_blocks.iter_nodes(), &mut result_visitor);
    for (place, location) in result_visitor.writes.iter() {
        let span = body.source_info(*location).span;
        tcx.sess
            .dcx()
            .struct_span_err(
                span,
                "Mutable argument must be used in Valid chunking or atomic functions".to_string(),
            )
            .emit();
    }
    let mut missing_sync: bool = false;

    for (place, state) in result_visitor.mem_state.iter() {
        if let SharedState::NeedSync(sync_scope, mut_loc, unsynced_loc) = state {
            let mut_span = body.source_info(*mut_loc).span;
            let mut unsynced_spans = FxHashSet::default();
            unsynced_loc.iter().for_each(|loc| {
                unsynced_spans.insert(body.source_info(*loc).span);
            });
            if unsynced_spans.is_empty() {
                // This is safe, as there is no read/write to the share mem origin after the write to the chunk.
                continue;
            }
            missing_sync = true;
            let mut err = tcx.sess.dcx().struct_span_err(
                mut_span,
                "The write needs a `sync_threads` called before other read/write".to_string(),
            );
            for span in unsynced_spans {
                err.span_note(span, "may need `sync_threads` before this read/write");
            }
            err.emit();
        }
    }

    if !result_visitor.writes.is_empty() {
        return Err(GpuCodegenError::MisuseMutableArgument);
    }
    if missing_sync {
        debug!("origins: {:?}", result_visitor.local_origin);
        return Err(GpuCodegenError::MissingSyncThreads);
    }
    Ok(())
}
