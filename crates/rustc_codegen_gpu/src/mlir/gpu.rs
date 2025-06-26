use melior::ir::operation::OperationBuilder;
use melior::ir::{Attribute, Identifier, Location, Operation, Type};

pub fn gpu_dim_op_index<'c>(
    ctx: &'c melior::Context,
    name: &str,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new(name, location)
        .add_attributes(&[(Identifier::new(ctx, "dimension"), dim)])
        .add_results(&[Type::index(ctx)])
        .build()
        .expect("valid operation")
}

pub fn thread_id<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    gpu_dim_op_index(ctx, "gpu.thread_id", dim, location)
}

pub fn global_id<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    gpu_dim_op_index(ctx, "gpu.global_id", dim, location)
}

pub fn block_dim<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    gpu_dim_op_index(ctx, "gpu.block_dim", dim, location)
}

pub fn grid_dim<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    gpu_dim_op_index(ctx, "gpu.grid_dim", dim, location)
}
