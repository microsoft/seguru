use rustc_data_structures::graph::DirectedGraph;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    Body, Local, Location, Operand, Place, Rvalue, Statement, StatementKind, Terminator,
    TerminatorKind,
};
use rustc_middle::ty::Ref;
use rustc_mir_dataflow::{Analysis, ResultsVisitor, visit_reachable_results};
use tracing::debug;

use crate::error::{GpuCodegenError, GpuCodegenResult};
use crate::scope::ScopedFun;

/// If the function is a kernel entry point, identify all mutable arguments.
/// Kernel entry functions are expected to use GlobalMem for mutable arguments.
/// Therefore, we verify whether these mutable arguments are actually mutated.
/// This check is a defense in depth measure, as the type system + trusted
/// gpu_macros should ideally prevent such issues.
/// However, the analysis is still useful for catching cases where mutable
/// arguments are used without using gpu_macros.
///
/// TODO: maybe we just need to check that a kernel entry function type never
/// use &mut T and only use GlobalMem<T>, &T or T: Copy whose size is smaller than 16 bytes?
struct MutArgAnalysis;

type Domain = DenseBitSet<Local>;

/// Find all mutable arguments and propagate their aliases through the forward analysis.
impl<'tcx> Analysis<'tcx> for MutArgAnalysis {
    // Track the mutable argument locals and their aliases.
    type Domain = DenseBitSet<Local>;
    type Direction = rustc_mir_dataflow::Forward;
    const NAME: &'static str = "MutArgAnalysis";

    fn bottom_value(&self, body: &Body<'tcx>) -> Self::Domain {
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        for local in body.args_iter() {
            let local_decl = &body.local_decls[local];
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
        if let StatementKind::Assign(lhs_and_rvalue) = &stmt.kind {
            let rvalue = &lhs_and_rvalue.1;
            match rvalue {
                Rvalue::Ref(_, _, src) | Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                    let lhs = &lhs_and_rvalue.0;
                    if state.contains(src.local) {
                        state.insert(lhs.local);
                    }
                }
                _ => {}
            }
        }
    }
}

/// MutArgDataFlowVistors detects invalid mutable argument used in non-chunking/atomic ops.
pub(crate) struct MutArgDataFlowVistors<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    local_decls: rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    // This is mostly not needed after represent &mut as GlobalMem, but keep it for now as a defense in depth.
    // Only used for global memory since shared memory is always protected by SharedMem type.
    invalid_write: Vec<(Place<'tcx>, Location)>,
}

impl<'tcx> MutArgDataFlowVistors<'tcx> {
    pub fn new(
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        local_decls: rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    ) -> Self {
        Self { tcx, invalid_write: Vec::new(), local_decls }
    }
}

/// ## The analysis logic for mutable argument access (global mem):
/// - Track the mutable argument locals and their aliases.
/// - If it is used for mutable argument to a trusted chunk function, it is allowed.
/// - If it is used for mutation in other cases, record the location and report error.
struct MutArgMirVisitor<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    state: &'a Domain,
    inside_fcall: bool,
    flow_visitor_res: &'a mut MutArgDataFlowVistors<'tcx>,
}

impl<'a, 'tcx> std::ops::Deref for MutArgMirVisitor<'a, 'tcx> {
    type Target = MutArgDataFlowVistors<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.flow_visitor_res
    }
}

impl<'a, 'tcx> std::ops::DerefMut for MutArgMirVisitor<'a, 'tcx> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.flow_visitor_res
    }
}

impl<'a, 'tcx> MutArgMirVisitor<'a, 'tcx> {
    fn is_mut_arg(&self, local: Local) -> bool {
        self.local_decls[local].mutability.is_mut() && !self.is_mut_ref_arg(local)
    }

    fn is_mut_ref_arg(&self, local: Local) -> bool {
        matches!(self.local_decls[local].ty.kind(), Ref(_, _, rustc_middle::ty::Mutability::Mut))
    }

    fn visit_place_for_invalid_write(
        &mut self,
        place: &Place<'tcx>,
        ctx: PlaceContext,
        loc: Location,
    ) {
        let mut write_accessed = false;
        match ctx {
            PlaceContext::MutatingUse(
                MutatingUseContext::Borrow | MutatingUseContext::RawBorrow,
            ) => {
                // Allow let a = &mut b since a is not used to mutate b yet.
            }
            PlaceContext::MutatingUse(_) => {
                // If projection is empty, it assigns value (e.g., let x =
                // &b[i];) to the local instead of dereferencing it (e.g.,
                // *x = 0).
                if !place.projection.is_empty() || self.is_mut_arg(place.local) {
                    debug!("Disallow mutate mutable argument: {:?} used as {:?}", place, ctx);
                    write_accessed = true;
                }
            }
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Inspect |
                NonMutatingUseContext::SharedBorrow | // e.g., let x = &b[i]; or let x = &b;
                NonMutatingUseContext::FakeBorrow|
                NonMutatingUseContext::RawBorrow |
                NonMutatingUseContext::PlaceMention |
                NonMutatingUseContext::Projection
             ) => {
                // let len = b.len() is not a write access.
            }
            PlaceContext::NonMutatingUse(NonMutatingUseContext::Move | NonMutatingUseContext::Copy) => {
                if !place.projection.is_empty() || self.inside_fcall || self.is_mut_arg(place.local)
                {
                    debug!(
                        "Disallow use mutable argument via index/function: {:?} used as {:?}",
                        place, ctx
                    );
                    write_accessed = true
                }
            }
            PlaceContext::NonUse(_) => {}
        }
        let is_mut_global = self.state.contains(place.local);
        if is_mut_global && write_accessed {
            self.invalid_write.push((*place, loc));
        }
    }
}

impl<'tcx> Visitor<'tcx> for MutArgMirVisitor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        self.visit_place_for_invalid_write(place, ctx, loc);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Visit the terminator to find reads/writes
        self.inside_fcall = true;
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                // Skip the chunk function to allow the trusted GPU memory partition.
                if let Some(gpu_item) = &attr.gpu_item {
                    if matches!(
                        ScopedFun::try_from(gpu_item.clone()),
                        Ok(ScopedFun::NewChunk(_) | ScopedFun::NewAtomic(_))
                    ) {
                        self.visit_span(*fn_span);
                        self.visit_operand(func, location);
                        // Skip the first argument (self.chunk_mut(...))
                        for arg in args.iter().skip(1) {
                            self.visit_operand(&arg.node, location);
                        }
                        self.visit_place(
                            destination,
                            PlaceContext::MutatingUse(MutatingUseContext::Call),
                            location,
                        );
                        self.inside_fcall = false;
                        return;
                    }
                }
            }
        }
        self.super_terminator(terminator, location);
        self.inside_fcall = false;
    }
}

impl<'tcx> ResultsVisitor<'tcx, MutArgAnalysis> for MutArgDataFlowVistors<'tcx> {
    fn visit_after_primary_statement_effect(
        &mut self,
        _results: &mut MutArgAnalysis,
        state: &Domain,
        stmt: &Statement<'tcx>,
        location: Location,
    ) {
        MutArgMirVisitor { tcx: self.tcx, state, flow_visitor_res: self, inside_fcall: false }
            .visit_statement(stmt, location);
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        _results: &mut MutArgAnalysis,
        state: &Domain,
        terminator: &Terminator<'tcx>,
        location: Location,
    ) {
        let mut visitor =
            MutArgMirVisitor { tcx: self.tcx, state, flow_visitor_res: self, inside_fcall: false };
        visitor.visit_terminator(terminator, location);
    }
}

/// Analyze the MIR body to check the mutable arguments.
pub(crate) fn analyze_mut_args<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> GpuCodegenResult<()> {
    let analysis = MutArgAnalysis;
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = MutArgDataFlowVistors::new(tcx, body.local_decls.clone());
    visit_reachable_results(body, &mut results.analysis, &results.results, &mut result_visitor);
    for (place, location) in result_visitor.invalid_write.iter() {
        let span = body.source_info(*location).span;
        tcx.sess
            .dcx()
            .struct_span_err(
                span,
                "Mutable argument must be used in Valid chunking or atomic functions".to_string(),
            )
            .emit();
    }

    if !result_visitor.invalid_write.is_empty() {
        return Err(GpuCodegenError::MisuseMutableArgument);
    }
    Ok(())
}
