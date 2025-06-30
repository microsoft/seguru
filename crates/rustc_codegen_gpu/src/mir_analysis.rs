use rustc_data_structures::fx::FxHashSet;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::pretty::write_mir_pretty;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    BasicBlock, Body, Local, Location, Operand, Place, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::Ref;
use rustc_mir_dataflow::{Analysis, Results, ResultsVisitor};

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

pub struct MutArgAliasVisitors<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    reads: Vec<(Place<'tcx>, Location)>,
    writes: Vec<(Place<'tcx>, Location)>,
}

impl<'tcx> MutArgAliasVisitors<'tcx> {
    pub fn new(tcx: rustc_middle::ty::TyCtxt<'tcx>) -> Self {
        Self { tcx, reads: Vec::new(), writes: Vec::new() }
    }
}
type Domain = DenseBitSet<Local>;

/// Visitor to collect reads/writes of mutable arguments.
/// It skips the chunk function to allow the trusted GPU memory partition.
struct ReadWriteVistor<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    state: &'a Domain,
    reads: &'a mut Vec<(Place<'tcx>, Location)>,
    writes: &'a mut Vec<(Place<'tcx>, Location)>,
}

impl<'tcx> Visitor<'tcx> for ReadWriteVistor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        if self.state.contains(place.local) {
            match ctx {
                PlaceContext::MutatingUse(..) => {
                    self.writes.push((*place, loc));
                }
                PlaceContext::NonMutatingUse(..) => {
                    self.reads.push((*place, loc));
                }
                _ => {}
            }
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Visit the terminator to find reads/writes
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                // Skip the chunk function to allow the trusted GPU memory partition.
                if let Some(crate::attr::GpuItem::IntoIter | crate::attr::GpuItem::SubsliceMut) = attr.gpu_item {
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
                    return;
                }
            }
        }
        self.super_terminator(terminator, location);
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
        ReadWriteVistor { tcx: self.tcx, state, reads: &mut self.reads, writes: &mut self.writes }
            .visit_statement(stmt, location);
    }

    fn visit_after_primary_terminator_effect(
        &mut self,
        _results: &mut Results<'tcx, MutArgAliasAnalysis>,
        state: &Domain,
        terminator: &'mir Terminator<'tcx>,
        location: Location,
    ) {
        ReadWriteVistor { tcx: self.tcx, state, reads: &mut self.reads, writes: &mut self.writes }
            .visit_terminator(terminator, location);
    }
}

/// Analyze the body to check the mutable argument locals and their uses.
/// TODO: check shared memory access.
fn analyze_access_to_mut<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    body: &Body<'tcx>,
) -> GpuCodegenResult<()> {
    let analysis = MutArgAliasAnalysis;
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = MutArgAliasVisitors::new(tcx);
    use rustc_data_structures::graph::DirectedGraph;
    results.visit_with(body, body.basic_blocks.iter_nodes(), &mut result_visitor);
    for (place, location) in result_visitor.reads.iter().chain(result_visitor.writes.iter()) {
        let span = body.source_info(*location).span;
        tcx.sess
            .dcx()
            .struct_span_err(span, "Mutable argument must be used in ChunkMut or Scope".to_string())
            .emit();
    }
    if result_visitor.reads.is_empty() && result_visitor.writes.is_empty() {
        Ok(())
    } else {
        Err(GpuCodegenError::MisuseMutableArgument)
    }
}

fn compute_loop_body(
    body: &Body<'_>,
    header: BasicBlock,
    backedge: BasicBlock,
) -> FxHashSet<BasicBlock> {
    let mut visited = FxHashSet::default();
    let mut stack = vec![backedge];

    while let Some(bb) = stack.pop() {
        if !visited.insert(bb) {
            continue;
        }
        if bb == header {
            continue;
        }
        for pred in body.basic_blocks.predecessors()[bb].iter().copied() {
            stack.push(pred);
        }
    }

    visited.insert(header); // include the header
    visited
}

// Might be useful to analyze loops and use scf instead of cf for optimization.
// Not fully implemented yet.
fn analyze_loop(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def_id: rustc_span::def_id::DefId,
) -> GpuCodegenResult<()> {
    let mir = tcx.optimized_mir(def_id);
    let dominators = mir.basic_blocks.dominators();
    let mut loop_pairs = vec![];
    for (bb, block) in mir.basic_blocks.iter_enumerated() {
        for succ in block.terminator().successors() {
            // Check if the successor is a backedge to the current block
            // If so, the successor is a loop header
            // The bb is the loop end.
            if dominators.dominates(succ, bb) {
                println!("Backedge from {:?} to {:?} (loop header)", bb, succ);
                let loop_body = compute_loop_body(mir, succ, bb);
                loop_pairs.push((bb, succ));
            }
        }
    }
    Ok(())
}

pub(crate) fn analyze_gpu_code(
    tcx: rustc_middle::ty::TyCtxt<'_>,
    def_id: rustc_span::def_id::DefId,
    is_kernel_entry: bool,
) -> GpuCodegenResult<()> {
    // Analyze the kernel to extract information like grid size, block size, etc.
    // This is a placeholder for actual analysis logic.
    let mir = tcx.optimized_mir(def_id);
    let mut out = Vec::new();
    write_mir_pretty(tcx, Some(def_id), &mut out).unwrap();
    if is_kernel_entry {
        analyze_access_to_mut(tcx, mir)?;
    }
    analyze_loop(tcx, def_id)
}
