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

struct MutArgAliasAnalysis;

/// Find all mutable argument locals and propagate their aliases through the forward analysis.
impl<'tcx> Analysis<'tcx> for MutArgAliasAnalysis {
    type Domain = DenseBitSet<Local>; // Or Place, if needed
    type Direction = rustc_mir_dataflow::Forward;
    const NAME: &'static str = "MutArgAliasAnalysis";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // Add initial mutable argument locals
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

#[derive(Debug)]
enum SharedState {
    NeedBlockSync(Location, Vec<Location>), // location where it needs sync_threads
    BlockSynced,
}

pub(crate) struct MutArgAliasVisitors<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    reads: Vec<(Place<'tcx>, Location)>,
    writes: Vec<(Place<'tcx>, Location)>,
    mem_state: FxHashMap<Local, SharedState>, // mem to its state
    chunk_map: FxHashMap<Local, SharedOrGlobal>, // Map from chunk local to global/shared local
    local_origin: FxHashMap<Local, Vec<Local>>, // Map from memory local to its original local
}

impl<'tcx> MutArgAliasVisitors<'tcx> {
    pub fn new(
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    ) -> Self {
        Self {
            tcx,
            reads: Vec::new(),
            writes: Vec::new(),
            local_decls,
            chunk_map: FxHashMap::default(),
            mem_state: FxHashMap::default(),
            local_origin: FxHashMap::default(),
        }
    }
}
type Domain = DenseBitSet<Local>;

#[allow(dead_code)]
enum SharedOrGlobal {
    Shared(Local), // is_shared, shared_var
    Global(Local),
}

/// Visitor to collect reads/writes of mutable arguments.
/// It skips the chunk function to allow the trusted GPU memory partition.
struct ReadWriteVistor<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    state: &'a Domain,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    reads: &'a mut Vec<(Place<'tcx>, Location)>,
    writes: &'a mut Vec<(Place<'tcx>, Location)>,
    chunk_map: &'a mut FxHashMap<Local, SharedOrGlobal>,
    mem_state: &'a mut FxHashMap<Local, SharedState>,
    called_sync_threads: bool,
    inside_fcall: bool,
    local_origin: &'a mut FxHashMap<Local, Vec<Local>>, // Map from memory local to its original local
}

impl<'a, 'tcx> ReadWriteVistor<'a, 'tcx> {
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
/// - When the shared chunk is mutated, marked the shared mem origin as NeedBlockSync, the current loc as mut_loc.
/// - If a place is used and its origin is in NeedBlockSync state, it means
///   we need a sync_thread between the mut_loc and the current loc.
impl<'tcx> Visitor<'tcx> for ReadWriteVistor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext};
        let origins = if let Some(origins) = self.local_origin.get(&place.local) {
            origins
        } else {
            &vec![place.local]
        };

        for origin in origins {
            if let Some(SharedState::NeedBlockSync(need_sync_loc, propose_loc)) =
                self.mem_state.get_mut(origin)
            {
                propose_loc.push(loc);
            }
        }

        if self.state.contains(place.local) {
            match ctx {
                PlaceContext::MutatingUse(
                    MutatingUseContext::Borrow | MutatingUseContext::RawBorrow,
                ) => {
                    // Borrowing a mutable argument is not allowed in GPU code.
                    // TODO: considering more allowed cases.
                    debug!("Allowing borrowing mutable argument: {:?} used as {:?}", place, ctx);
                }
                PlaceContext::MutatingUse(_) => {
                    // If projection is empty, it is assign value (e.g.,
                    // let x = &b[i];) to the local instead of dereferencing it
                    // (e.g., *x = 0).
                    if !place.projection.is_empty() || self.is_mut_arg(place.local) {
                        debug!("Disallow use mutable argument: {:?} used as {:?}", place, ctx);
                        self.writes.push((*place, loc));
                    } else {
                        debug!("Allowing mutable argument: {:?} used as {:?}", place, ctx);
                    }
                }
                PlaceContext::NonMutatingUse(
                    NonMutatingUseContext::Move | NonMutatingUseContext::Copy,
                ) => {
                    if !place.projection.is_empty()
                        || self.inside_fcall
                        || self.is_mut_arg(place.local)
                    {
                        debug!("Disallow use mutable argument: {:?} used as {:?}", place, ctx);
                        self.reads.push((*place, loc));
                    } else {
                        debug!("Checking non-mutating use of mutable: {:?} at {:?}", place, loc);
                        debug!("Allowing non-mutating use of mutable: {:?} as {:?}", place, ctx);
                    }
                }
                _ => {}
            }
        }
        // If the place is a chunked variable, update its shared memory state.
        // If the operation is a write, mark it as NeedBlockSync.
        if self.chunk_map.contains_key(&place.local) {
            if let SharedOrGlobal::Shared(shared_var) = self.chunk_map[&place.local] {
                if let PlaceContext::MutatingUse(_) = ctx {
                    let origins = if let Some(origins) = self.local_origin.get(&shared_var) {
                        origins
                    } else {
                        &vec![shared_var]
                    };
                    for origin in origins {
                        debug!(
                            "Marking shared var {:?} as NeedBlockSync: {:?}, {:?}",
                            origin, place.local, loc
                        );
                        self.mem_state.insert(*origin, SharedState::NeedBlockSync(loc, vec![]));
                    }
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
        let mut is_shared_mem = false;
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                // Skip the chunk function to allow the trusted GPU memory partition.
                let mut is_trusted_chunk_function = false;
                if attr.gpu_item == Some(GpuItem::SyncThreads) {
                    self.called_sync_threads = true;
                }
                match &attr.gpu_item {
                    Some(GpuItem::DiagnoseOnly(name))
                        if name == "gpu::chunk_mut" || name == "gpu::shared_chunk_mut" =>
                    {
                        is_shared_mem = name == "gpu::shared_chunk_mut";
                        is_trusted_chunk_function = true;
                    }
                    Some(
                        GpuItem::SubsliceMut
                        | GpuItem::NewChunk
                        | GpuItem::AtomicRMW
                        | GpuItem::GetLocalMut2D,
                    ) => {
                        is_trusted_chunk_function = true;
                    }
                    _ => {}
                }
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
                    self.chunk_map.insert(
                        chunked_var,
                        if is_shared_mem {
                            SharedOrGlobal::Shared(global_or_shared_var)
                        } else {
                            SharedOrGlobal::Global(global_or_shared_var)
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

impl<'mir, 'tcx> ResultsVisitor<'mir, 'tcx, MutArgAliasAnalysis> for MutArgAliasVisitors<'tcx> {
    fn visit_after_primary_statement_effect(
        &mut self,
        _results: &mut Results<'tcx, MutArgAliasAnalysis>,
        state: &Domain,
        stmt: &'mir Statement<'tcx>,
        location: Location,
    ) {
        ReadWriteVistor {
            tcx: self.tcx,
            local_decls: self.local_decls,
            state,
            reads: &mut self.reads,
            writes: &mut self.writes,
            chunk_map: &mut self.chunk_map,
            mem_state: &mut self.mem_state,
            called_sync_threads: false,
            inside_fcall: false,
            local_origin: &mut self.local_origin,
        }
        .visit_statement(stmt, location);
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        _results: &mut Results<'tcx, MutArgAliasAnalysis>,
        state: &Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) {
        let mut visitor = ReadWriteVistor {
            tcx: self.tcx,
            local_decls: self.local_decls,
            state,
            reads: &mut self.reads,
            writes: &mut self.writes,
            chunk_map: &mut self.chunk_map,
            mem_state: &mut self.mem_state,
            called_sync_threads: false,
            inside_fcall: false,
            local_origin: &mut self.local_origin,
        };
        visitor.visit_terminator(terminator, location);

        // If sync_threads is called, all chunk states become BlockSynced.
        if visitor.called_sync_threads {
            for (local, state) in self.mem_state.iter_mut() {
                debug!("Marking shared var {:?} as synced: {:?}", local, location);
                *state = SharedState::BlockSynced;
            }
        }
    }
}

/// Analyze the body to check the mutable argument locals and their uses.
/// TODO: check shared memory access.
pub(crate) fn analyze_access_to_mut<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &'tcx Body<'tcx>,
) -> GpuCodegenResult<()> {
    let analysis = MutArgAliasAnalysis;
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = MutArgAliasVisitors::new(tcx, &body.local_decls);
    results.visit_with(body, body.basic_blocks.iter_nodes(), &mut result_visitor);
    for (place, location) in result_visitor.reads.iter().chain(result_visitor.writes.iter()) {
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
        if let SharedState::NeedBlockSync(mut_loc, unsynced_loc) = state {
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

    if !result_visitor.reads.is_empty() || !result_visitor.writes.is_empty() {
        return Err(GpuCodegenError::MisuseMutableArgument);
    }
    if missing_sync {
        debug!("origins: {:?}", result_visitor.local_origin);
        return Err(GpuCodegenError::MissingSyncThreads);
    }
    Ok(())
}
