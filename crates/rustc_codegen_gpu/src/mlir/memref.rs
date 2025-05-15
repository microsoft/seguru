use melior::ir::{
    operation::OperationBuilder, Location, Operation, Type,
    Value,
};

#[allow(dead_code)]
pub fn reinterpret_cast<'c>(
    ty: Type<'c>,
    ptr: Value<'c, '_>,
    indices: &[Value<'c, '_>],
    location: Location<'c>,
) -> Operation<'c> {
    OperationBuilder::new("memref.reinterpret_cast", location)
        .add_operands(&[ptr])
        .add_results(&[ty])
        .build()
        .expect("valid operation")
}
