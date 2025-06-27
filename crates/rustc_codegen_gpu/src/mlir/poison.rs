/// Tuple is a virtual type.
use melior::ir::Type;
use melior::ir::attribute::StringAttribute;
use melior::ir::operation::OperationBuilder;
use melior::ir::{Attribute, Identifier, Operation};

pub fn const_poison<'ml>(
    mlir_ctx: &'ml crate::mlir::Context,
    ty: Type<'ml>,
    location: melior::ir::Location<'ml>,
) -> Operation<'ml> {
    OperationBuilder::new("arith.constant", location)
        .add_attributes(&[
            (
                Identifier::new(mlir_ctx, "to_remove"),
                StringAttribute::new(mlir_ctx, "poison").into(),
            ),
            (Identifier::new(mlir_ctx, "value"), Attribute::parse(mlir_ctx, "0: i64").unwrap()),
        ])
        .add_results(&[ty])
        .build()
        .expect("valid operation")
}
