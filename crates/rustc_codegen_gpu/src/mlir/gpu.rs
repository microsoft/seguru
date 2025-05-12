use melior::{
    dialect::ods::gpu,
    ir::{
        operation::OperationBuilder, r#type::MemRefType, Attribute, Identifier, Location,
        Operation, Type, Value,
    },
};

pub fn thread_id<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("gpu.thread_id", location)
        .add_attributes(&[(Identifier::new(ctx, "dimension"), dim)])
        .add_results(&[Type::index(ctx)])
        .build()
        .expect("valid operation")
}

pub fn global_id<'c>(
    ctx: &'c melior::Context,
    dim: Attribute<'c>,
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("gpu.global_id", location)
        .add_attributes(&[(Identifier::new(ctx, "dimension"), dim)])
        .add_results(&[Type::index(ctx)])
        .build()
        .expect("valid operation")
}
