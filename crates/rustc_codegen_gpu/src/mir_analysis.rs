use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::graph::DirectedGraph;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::pretty::write_mir_pretty;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{
    BasicBlock, Body, Local, Location, Operand, Place, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::{Instance, Ref, TyCtxt, TyKind};
use rustc_mir_dataflow::{Analysis, Results, ResultsVisitor};
use tracing::debug;

use crate::attr::{GpuAttributes, GpuItem};
use crate::error::{GpuCodegenError, GpuCodegenResult};

fn from_inlined_func<'tcx>(
    body: &Body<'tcx>,
    source_info: &rustc_middle::mir::SourceInfo,
) -> Option<Instance<'tcx>> {
    let stmt_scope = source_info.scope;
    let scope_data = &body.source_scopes[stmt_scope];

    scope_data.inlined.map(|i| i.0)
}

/// Return the (inlined function, real_function)
fn check_terminator_call<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    terminator: &Terminator<'tcx>,
) -> (Option<GpuAttributes>, Option<GpuAttributes>) {
    let mut inlines = None;
    let mut real = None;
    if let TerminatorKind::Call { func, args, .. } = &terminator.kind {
        // Check if the function being called is an inlined function
        if let Some(inlined_info) = from_inlined_func(body, &terminator.source_info) {
            let gpu_attr = crate::attr::GpuAttributes::build(&tcx, inlined_info.def_id());
            inlines = Some(gpu_attr);
        }
        if let Some((def_id, generic_args)) = func.const_fn_def() {
            if let Ok(Some(instance)) =
                Instance::try_resolve(tcx, body.typing_env(tcx), def_id, generic_args)
            {
                let gpu_attr = crate::attr::GpuAttributes::build(&tcx, instance.def_id());
                real = Some(gpu_attr);
            }
        }
    }
    (inlines, real)
}

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

pub struct MutArgAliasVisitors<'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    reads: Vec<(Place<'tcx>, Location)>,
    writes: Vec<(Place<'tcx>, Location)>,
}

impl<'tcx> MutArgAliasVisitors<'tcx> {
    pub fn new(
        tcx: rustc_middle::ty::TyCtxt<'tcx>,
        local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    ) -> Self {
        Self { tcx, reads: Vec::new(), writes: Vec::new(), local_decls }
    }
}
type Domain = DenseBitSet<Local>;

/// Visitor to collect reads/writes of mutable arguments.
/// It skips the chunk function to allow the trusted GPU memory partition.
struct ReadWriteVistor<'a, 'tcx> {
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    state: &'a Domain,
    local_decls: &'tcx rustc_index::IndexVec<Local, rustc_middle::mir::LocalDecl<'tcx>>,
    reads: &'a mut Vec<(Place<'tcx>, Location)>,
    writes: &'a mut Vec<(Place<'tcx>, Location)>,
    inside_fcall: bool,
}

impl<'a, 'tcx> ReadWriteVistor<'a, 'tcx> {
    fn is_mut_arg(&self, local: Local) -> bool {
        self.local_decls[local].mutability.is_mut() && !self.is_mut_ref_arg(local)
    }

    fn is_mut_ref_arg(&self, local: Local) -> bool {
        matches!(self.local_decls[local].ty.kind(), Ref(_, _, rustc_middle::ty::Mutability::Mut))
    }
}

impl<'tcx> Visitor<'tcx> for ReadWriteVistor<'_, 'tcx> {
    fn visit_place(&mut self, place: &Place<'tcx>, ctx: PlaceContext, loc: Location) {
        use rustc_middle::mir::visit::{MutatingUseContext, NonMutatingUseContext};
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
                        debug!("Allowing non-mutating use of mutable: {:?} as {:?}", place, ctx);
                    }
                }
                _ => {}
            }
        }
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        // Visit the terminator to find reads/writes
        self.inside_fcall = true;
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                // Skip the chunk function to allow the trusted GPU memory partition.
                let mut is_trusted_chunk_function = false;
                match &attr.gpu_item {
                    Some(GpuItem::DiagnoseOnly(name)) if name == "gpu::chunk_mut" => {
                        is_trusted_chunk_function = true;
                    }
                    Some(
                        GpuItem::SubsliceMut
                        | GpuItem::NewChunk
                        | GpuItem::AtomicAdd
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
            inside_fcall: false,
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
        ReadWriteVistor {
            tcx: self.tcx,
            local_decls: self.local_decls,
            state,
            reads: &mut self.reads,
            writes: &mut self.writes,
            inside_fcall: false,
        }
        .visit_terminator(terminator, location);
    }
}

/// Visitor to detect if a specific local is assigned from a GPU-specific intrinsic.
pub struct TaintSourceDetector<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub sources: DenseBitSet<Local>,
}

impl<'tcx> Visitor<'tcx> for TaintSourceDetector<'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call { func, destination, .. } = &terminator.kind {
            if let Some((def_id, generic_args)) = func.const_fn_def() {
                let attr = crate::attr::GpuAttributes::build(&self.tcx, def_id);
                debug!(
                    "Checking terminator at {:?} {:?} for taint source: {:?} {:?}",
                    location, def_id, attr.gpu_item, attr.ret_sync_data
                );

                // Those functions are not considered as source since they should
                // generate non-diversed data is the input is non-diversed.
                // TODO: confirm all DeviceIntrinsic is non-diversed.
                // Currently DeviceIntrinsic only defines some float compute operations.
                // We only care the returned value since the mut arg will
                // only be tainted when reaching that the terminator.
                let trusted_non_diversed =
                    matches!(
                        attr.gpu_item,
                        Some(
                            GpuItem::BlockDim
                                | GpuItem::GridDim
                                | GpuItem::AllReduce
                                | GpuItem::SubgroupReduce
                                | GpuItem::SubgroupSize
                                | GpuItem::DynamicShared
                                | GpuItem::NewSharedMem
                                | GpuItem::DeviceIntrinsic(_)
                        )
                    ) || attr.ret_sync_data.contains(&GpuAttributes::MAX_FN_IN_PARAMS);

                // Since any device function may call thread_id, block_id,
                // we conservatively consider all device function calls as taint sources
                // except for the trusted non-diversed functions.
                if matches!(attr.gpu_item, Some(GpuItem::ThreadId) | Some(GpuItem::BlockId))
                    || (!trusted_non_diversed && attr.device)
                {
                    // Check if the destination is the target local
                    self.sources.insert(destination.local);
                }
            }
        }

        self.super_terminator(terminator, location);
    }
}

/// Tracking the diversed var that might has different values across threads.
struct TaintTracking<'tcx> {
    kernel_entry: bool,
    init: DenseBitSet<Local>,
    implicit: DenseBitSet<rustc_middle::mir::BasicBlock>,
    tcx: TyCtxt<'tcx>,
    body: &'tcx Body<'tcx>,
}

fn operand_local<'tcx>(op: &Operand<'tcx>) -> Option<Local> {
    match op {
        Operand::Copy(place) | Operand::Move(place) => Some(place.local),
        Operand::Constant(_) => None,
    }
}

impl<'tcx> TaintTracking<'tcx> {
    pub fn new(
        kernel_entry: bool,
        tainted: DenseBitSet<Local>,
        tcx: TyCtxt<'tcx>,
        body: &'tcx Body<'tcx>,
    ) -> Self {
        Self {
            kernel_entry,
            init: tainted,
            implicit: DenseBitSet::new_empty(body.basic_blocks.len()),
            tcx,
            body,
        }
    }

    fn propogate_fn_call(
        &self,
        args: &[rustc_span::source_map::Spanned<Operand<'tcx>>],
        destination: Option<&Place<'tcx>>,
        attr: &Option<GpuAttributes>,
        state: &mut DenseBitSet<Local>,
    ) {
        let mut mut_args = vec![];
        let mut tainted = false;
        args.iter().enumerate().for_each(|(i, arg)| {
            if let Some(local) = operand_local(&arg.node) {
                let local_decl = &self.body.local_decls[local];
                if local_decl.mutability.is_mut() {
                    mut_args.push((i, local));
                }
            }
        });
        args.iter().for_each(|arg| {
            if let Some(local) = operand_local(&arg.node) {
                if state.contains(local) {
                    tainted = true;
                }
            }
        });

        // Even if the function is not tainted, we still need to
        // propagate the tainted locals to the mut args.
        // This is because the function body may call functions
        // that could create taint source.
        // No need to deal with destination since TaintSourceDetector
        // will handle the destination local.
        for (i, arg) in mut_args.iter() {
            if let Some(attr) = attr {
                // If the function is marked as ret_sync_data,
                // we should not propagate the taint to the argument.
                // This is because the function is expected to return
                // the sync data and thus the argument should not be tainted.
                // TODO: we will check and verify that the function is indeed
                // returning the sync data.
                if attr.ret_sync_data.contains(i) {
                    continue;
                }
            }
            state.insert(*arg);
        }

        // If the function includes a taint source, we aggresively
        // propagate the taint to the destination and mut args.
        if tainted {
            if let Some(destination) = destination {
                state.insert(destination.local);
            }
            for (i, arg) in mut_args.iter() {
                state.insert(*arg);
            }
        }
    }
}

impl<'tcx> Analysis<'tcx> for TaintTracking<'tcx> {
    type Domain = DenseBitSet<Local>;
    type Direction = rustc_mir_dataflow::Forward;
    const NAME: &'static str = "TaintTracking";

    fn bottom_value(&self, body: &Body<'_>) -> Self::Domain {
        DenseBitSet::new_empty(body.local_decls.len())
    }

    fn initialize_start_block(&self, body: &Body<'tcx>, state: &mut Self::Domain) {
        // Initialize tainted locals
        let mut source = TaintSourceDetector { tcx: self.tcx, sources: state.clone() };

        source.visit_body(body);

        *state = self.init.clone();

        for local in source.sources.iter() {
            state.insert(local);
        }

        if self.kernel_entry {
            debug!("TaintTracking(kernel) initialized with {:?} tainted locals", state);
            return;
        }

        // The inputs to a dev func might be diversed across different threads
        for local in body.args_iter() {
            state.insert(local);
        }

        debug!("TaintTracking initialized with {:?} tainted locals", state);
    }

    fn apply_primary_statement_effect(
        &mut self,
        state: &mut Domain,
        statement: &Statement<'tcx>,
        _location: Location,
    ) {
        if let StatementKind::Assign(box (lhs, rvalue)) = &statement.kind {
            let mut tainted = false;
            // Walk all operands used in the rvalue
            match rvalue {
                Rvalue::Use(op)
                | Rvalue::Repeat(op, _)
                | Rvalue::UnaryOp(_, op)
                | Rvalue::Cast(_, op, _)
                | Rvalue::ShallowInitBox(op, _)
                | Rvalue::WrapUnsafeBinder(op, _) => {
                    if let Some(local) = operand_local(op) {
                        if state.contains(local) {
                            tainted = true;
                        }
                    }
                }

                Rvalue::BinaryOp(_, box (op1, op2)) => {
                    if operand_local(op1).is_some_and(|l| state.contains(l))
                        || operand_local(op2).is_some_and(|l| state.contains(l))
                    {
                        tainted = true;
                    }
                }

                Rvalue::Ref(_, _, place) | Rvalue::CopyForDeref(place) => {
                    if state.contains(place.local) {
                        tainted = true;
                    }
                }

                Rvalue::Aggregate(_, fields) => {
                    for op in fields {
                        if let Some(local) = operand_local(op) {
                            if state.contains(local) {
                                tainted = true;
                                break;
                            }
                        }
                    }
                }

                Rvalue::Discriminant(place) | Rvalue::Len(place) | Rvalue::RawPtr(_, place) => {
                    tainted = state.contains(place.local);
                }

                Rvalue::NullaryOp(..) => {}

                Rvalue::ThreadLocalRef(_) => {
                    panic!("Thread-local references are not supported in GPU code");
                }
            }

            if tainted {
                state.insert(lhs.local);
            }
        }
    }

    fn apply_primary_terminator_effect<'mir>(
        &mut self,
        state: &mut Domain,
        terminator: &'mir Terminator<'tcx>,
        _location: Location,
    ) -> rustc_middle::mir::TerminatorEdges<'mir, 'tcx> {
        let dominators = self.body.basic_blocks.dominators();
        // Taint the destination and mut arg if args include a tainted local.
        match &terminator.kind {
            TerminatorKind::Call { func, ref args, destination, .. } => {
                let (_, gpu_attr) = check_terminator_call(self.tcx, self.body, terminator);
                self.propogate_fn_call(args, Some(destination), &gpu_attr, state);
            }
            TerminatorKind::TailCall { func, ref args, .. } => {
                let (_, gpu_attr) = check_terminator_call(self.tcx, self.body, terminator);
                self.propogate_fn_call(args, None, &gpu_attr, state);
            }

            TerminatorKind::SwitchInt { ref targets, ref discr, .. } => {
                // If the discriminant is tainted, all targets are tainted.
                if let Some(local) = operand_local(discr) {
                    if state.contains(local) {
                        for target in targets.all_targets() {
                            debug!("Setting block {:?} as implicit", target);
                            self.implicit.insert(*target);
                        }
                        let mut intersect: Option<DenseBitSet<BasicBlock>> = None;
                        let mut union = DenseBitSet::new_empty(self.body.basic_blocks.len());
                        for target in targets.all_targets() {
                            let reachable = reachable_bb(self.body, *target);
                            if let Some(intersect) = intersect.as_mut() {
                                intersect.intersect(&reachable);
                            } else {
                                intersect = Some(reachable.clone());
                            }
                            union.union(&reachable);
                        }

                        // We remove the bb if the bb is reachable from all
                        // targets which means it does not depend on the
                        // discriminant.
                        union.subtract(intersect.as_ref().unwrap());

                        self.implicit.union(&union);
                        debug!("Marking blocks {:?} as implicit", union);
                        /*
                        // The union should cover the following strict cases.
                        for bb in self.body.basic_blocks.iter_nodes() {
                            for target in targets.all_targets() {
                                // If bb is only reachable from one target,
                                if dominators.dominates(*target, bb) {
                                    debug!("Marking block {:?} as implicit", bb);
                                    self.implicit.insert(bb);
                                    break;
                                }
                            }
                        }
                        */
                    }
                }
            }
            TerminatorKind::InlineAsm { ref operands, .. } => {
                // Inline assembly can have side effects, so we assume all outputs are tainted.
                let mut tainted = false;
                let mut out = vec![];
                for op in operands.iter() {
                    match op {
                        rustc_middle::mir::InlineAsmOperand::In { value, .. } => {
                            if let Some(local) = operand_local(value) {
                                tainted = state.contains(local);
                            }
                        }
                        rustc_middle::mir::InlineAsmOperand::Out { place: Some(place), .. } => {
                            out.push(place.local);
                        }
                        rustc_middle::mir::InlineAsmOperand::InOut {
                            in_value, out_place, ..
                        } => {
                            if let Some(local) = operand_local(in_value) {
                                tainted = state.contains(local);
                            }
                            if let Some(place) = out_place {
                                out.push(place.local);
                            }
                        }
                        _ => {}
                    }
                }
                if tainted {
                    for local in out {
                        state.insert(local);
                    }
                }
            }
            TerminatorKind::Yield { ref value, resume, resume_arg, .. } => {
                // If the yield value is a local, check if it is tainted.
                if let Some(local) = operand_local(value) {
                    if state.contains(local) {
                        state.insert(resume_arg.local);
                        for stmt in &self.body.basic_blocks[*resume].statements {
                            if let StatementKind::Assign(box (lhs, _)) = &stmt.kind {
                                state.insert(lhs.local);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
        terminator.edges()
    }
}

/// Returns true if `target` is reachable from `start` block
fn reachable_bb<'tcx>(body: &Body<'tcx>, start: BasicBlock) -> DenseBitSet<BasicBlock> {
    let mut visited = DenseBitSet::new_empty(body.basic_blocks.len());
    let mut worklist = vec![start];

    while let Some(bb) = worklist.pop() {
        if visited.contains(bb) {
            continue;
        }

        visited.insert(bb);

        let successors = body.basic_blocks[bb].terminator().successors();
        worklist.extend(successors);
    }

    visited
}

struct TaintResultVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'tcx Body<'tcx>,
    invalid_diversed: Vec<rustc_span::Span>,
}

impl<'tcx> TaintResultVisitor<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, body: &'tcx Body<'tcx>) -> Self {
        Self { tcx, body, invalid_diversed: Vec::new() }
    }
}

fn location_span<'tcx>(body: &Body<'tcx>, location: Location) -> rustc_span::Span {
    let block = &body.basic_blocks[location.block];

    let source_info = if location.statement_index < block.statements.len() {
        // Statement at this location

        block.statements[location.statement_index].source_info
    } else {
        // Refers to the terminator of the block
        block.terminator().source_info
    };

    let mut span = source_info.span;

    // If the source info is inlined, we need to find the original span
    // This is necessary to get the correct span for inlined functions.
    let mut scope = source_info.scope;
    while let Some(s) = body.source_scopes[scope].inlined_parent_scope {
        scope = s;
        span = body.source_scopes[scope].inlined.unwrap().1;
    }
    span
}

impl<'mir, 'tcx> ResultsVisitor<'mir, 'tcx, TaintTracking<'tcx>> for TaintResultVisitor<'tcx> {
    /// We only visit early terminator since the data that is impacted by
    /// the current and future  statement/terminator does not matter.
    fn visit_after_early_terminator_effect(
        &mut self,
        results: &mut Results<'tcx, TaintTracking<'tcx>>,
        state: &DenseBitSet<Local>,
        terminator: &Terminator<'tcx>,
        location: Location,
    ) {
        debug!("Visiting terminator at {:?} with state: {:?}", location, state);
        if let TerminatorKind::Call { func, destination, ref args, fn_span, .. } = &terminator.kind
        {
            let mut sync_data = None;
            let (inline, real) = check_terminator_call(self.tcx, self.body, terminator);
            if let Some(real_attr) = real {
                sync_data = real_attr.sync_data;
            }

            let span = location_span(self.body, location);

            if let Some(sync_data) = sync_data {
                // Check the arguments that require to be used in a non-diversed
                // way.
                for idx in sync_data {
                    let arg = &args[idx];
                    if let Some(local) = operand_local(&arg.node) {
                        if state.contains(local) {
                            debug!(
                                "Invalid use of diversed data in GPU code at {:?} for local {:?}",
                                span, local
                            );
                            self.invalid_diversed.push(arg.span);
                            self.invalid_diversed.push(span);
                        }
                    }
                }

                // If this terminator is control-dependent on a tainted value
                // (i.e., part of an implicit flow), then it is unsafe to use
                // divergent or data-dependent values here.
                for bb in results.analysis.implicit.iter() {
                    if self.body.basic_blocks[bb].terminator().source_info == terminator.source_info
                    {
                        self.invalid_diversed.push(span);
                    }
                }
            }
        }
    }
}

/// Analyze the body to check use of chunk function and shared variable.
/// Ensures that the args to those functions are used in a non-diversed way.
fn analyze_diversed_data<'tcx>(
    tcx: rustc_middle::ty::TyCtxt<'tcx>,
    def_id: rustc_span::def_id::DefId,
    body: &'tcx Body<'tcx>,
    kernel_entry: bool,
) -> GpuCodegenResult<()> {
    let analysis = TaintTracking::new(
        kernel_entry, // kernel entry
        DenseBitSet::new_empty(body.local_decls.len()),
        tcx,
        body,
    );
    let mut results = analysis.iterate_to_fixpoint(tcx, body, None);
    let mut result_visitor = TaintResultVisitor::new(tcx, body);
    results.visit_with(body, body.basic_blocks.iter_nodes(), &mut result_visitor);
    for span in &result_visitor.invalid_diversed {
        tcx.sess
            .dcx()
            .struct_span_err(
                *span,
                format!("Invalid use of diversed data in GPU code at {:?}", def_id),
            )
            .emit();
    }
    if result_visitor.invalid_diversed.is_empty() {
        Ok(())
    } else {
        Err(GpuCodegenError::InvalidDiversedData)
    }
}

/// Analyze the body to check the mutable argument locals and their uses.
/// TODO: check shared memory access.
fn analyze_access_to_mut<'tcx>(
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

fn is_fn_unsafe(tcx: rustc_middle::ty::TyCtxt<'_>, def_id: rustc_span::def_id::DefId) -> bool {
    // Check that the item is a function-like item
    let ty = tcx.type_of(def_id).skip_binder();

    let poly_sig = match ty.kind() {
        TyKind::Closure(_, substs) => {
            // Extract closure's internal function signature
            substs.as_closure().sig()
        }
        TyKind::FnDef(..) => {
            // Standard function (free fn, method, etc.)
            tcx.fn_sig(def_id).skip_binder()
        }
        _ => {
            panic!("DefId does not refer to a callable item: {:?}", ty)
        }
    };

    poly_sig.skip_binder().safety == rustc_hir::Safety::Unsafe
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
    debug!("MIR for {:?}:\n{}", def_id, String::from_utf8_lossy(&out));
    let mut skip_check = is_fn_unsafe(tcx, def_id);
    let crate_name = tcx.crate_name(def_id.krate);
    if crate_name.as_str() == "gpu" {
        debug!("Skipping GPU crate checks {:?} {}", def_id, crate_name);
        skip_check = true; // Skip checks for the gpu crate
    }
    if is_kernel_entry && !skip_check {
        analyze_access_to_mut(tcx, mir)?;
    }
    if !skip_check {
        debug!("analyze_diversed_data {:?}", def_id);
        analyze_diversed_data(tcx, def_id, mir, is_kernel_entry)?;
    }
    analyze_loop(tcx, def_id)
}
