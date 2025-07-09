use melior::ir::attribute::{BoolAttribute, IntegerAttribute};
use melior::ir::operation::OperationBuilder;
use melior::ir::r#type::IntegerType;
use melior::ir::{Attribute, Identifier, Location, Operation, Type, Value, ValueLike};

pub enum DimFn {
    GlobalThreadId,
    ThreadId,
    BlockId,
    BlockDim,
    GridDim,
}

pub enum NonDimFn {
    SubgroupId,
    SubgroupSize,
    LaneId,
}

impl DimFn {
    fn name(&self) -> &'static str {
        match self {
            DimFn::GlobalThreadId => "gpu.global_id",
            DimFn::ThreadId => "gpu.thread_id",
            DimFn::BlockId => "gpu.block_id",
            DimFn::BlockDim => "gpu.block_dim",
            DimFn::GridDim => "gpu.grid_dim",
        }
    }

    pub fn build<'c>(
        &self,
        ctx: &'c melior::Context,
        dim: Attribute<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new(self.name(), location)
            .add_attributes(&[(Identifier::new(ctx, "dimension"), dim)])
            .add_results(&[Type::index(ctx)])
            .build()
            .expect("valid operation")
    }
}

impl NonDimFn {
    fn name(&self) -> &'static str {
        match self {
            NonDimFn::SubgroupId => "gpu.subgroup_id",
            NonDimFn::SubgroupSize => "gpu.subgroup_size",
            NonDimFn::LaneId => "gpu.lane_id",
        }
    }

    pub fn build<'c>(&self, ctx: &'c melior::Context, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new(self.name(), location)
            .add_results(&[Type::index(ctx)])
            .build()
            .expect("valid operation")
    }
}

pub(crate) fn all_reduce<'ml, 'a>(
    ctx: &'ml melior::Context,
    arg: Value<'ml, 'a>,
    op_attr: Attribute<'ml>,
    uniform: bool,
    loc: Location<'ml>,
) -> Operation<'ml> {
    let region = melior::ir::Region::new();
    let builder = melior::dialect::ods::gpu::AllReduceOperationBuilder::new(ctx, loc)
        .value(arg)
        .body(region)
        .op(op_attr);
    let builder = if uniform { builder.uniform(Attribute::unit(ctx)) } else { builder };
    builder.build().into()
}

/// If uniform is true, it requires all threads in the subgroup to perform the same operation or perform no operation at all.
/// If uniform is false, it allows threads in the subgroup to perform different operations, e.g., reduce under some condition.
pub(crate) fn subgroup_reduce<'ml, 'a>(
    ctx: &'ml melior::Context,
    arg: Value<'ml, 'a>,
    op_attr: Attribute<'ml>,
    uniform: bool,
    cluster_size: usize,
    cluster_stride: usize,
    loc: Location<'ml>,
) -> Operation<'ml> {
    assert!(cluster_size == 32);
    assert!(cluster_stride == 1);
    assert!(uniform);
    let type_signless_i32 = Type::from(IntegerType::new(ctx, 32));
    let cluster_size = IntegerAttribute::new(type_signless_i32, cluster_size as i64);
    let cluster_stride = IntegerAttribute::new(type_signless_i32, cluster_stride as i64);
    let builder = melior::dialect::ods::gpu::SubgroupReduceOperationBuilder::new(ctx, loc)
        .value(arg)
        .op(op_attr);
    //.cluster_size(cluster_size)
    //.cluster_stride(cluster_stride);
    let builder = if uniform { builder.uniform(Attribute::unit(ctx)) } else { builder };
    builder.build().into()
}

pub(crate) fn nvvm_redux_sync<'ml, 'a>(
    ctx: &'ml melior::Context,
    arg: Value<'ml, 'a>,
    mask: Value<'ml, 'a>,
    op_attr: Attribute<'ml>,
    abs: bool,
    nan: bool,
    location: Location<'ml>,
) -> Operation<'ml> {
    OperationBuilder::new("nvvm.redux.sync", location)
        .add_attributes(&[
            (Identifier::new(ctx, "kind"), op_attr),
            (Identifier::new(ctx, "abs"), BoolAttribute::new(ctx, abs).into()),
            (Identifier::new(ctx, "nan"), BoolAttribute::new(ctx, nan).into()),
        ])
        .add_operands(&[arg, mask])
        .add_results(&[arg.r#type()])
        .build()
        .expect("valid operation")
}
