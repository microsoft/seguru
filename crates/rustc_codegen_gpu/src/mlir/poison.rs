use melior::ir::attribute::{IntegerAttribute, StringAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike, OperationResult};
use melior::ir::r#type::{IntegerType, TupleType};
use melior::ir::{Identifier, Operation, TypeLike, ValueLike};
/// Tuple is a virtual type.
use melior::ir::{Type, Value};
use tracing::debug;

pub fn const_poison<'ml>(
    mlir_ctx: &'ml crate::mlir::Context,
    ty: Type<'ml>,
    location: melior::ir::Location<'ml>,
) -> Operation<'ml> {
    OperationBuilder::new("poison.const", location)
        .add_attributes(&[(
            Identifier::new(mlir_ctx, "to_remove"),
            StringAttribute::new(mlir_ctx, "poison").into(),
        )])
        .add_results(&[ty])
        .build()
        .expect("valid operation")
}

/// insert_value is always used after const_poison in immediate_or_packed_pair at rustc_codegen_ssa/src/mir/operand.rs
/// so it is safe to assume that the decoded value is always a tuple and all fields are inserted by insert_value.
pub fn insert_value<'ml, 'a>(
    mlir_ctx: &'ml crate::mlir::Context,
    agg_val: Value<'ml, 'a>,
    elt: Value<'ml, 'a>,
    idx: u64,
    location: melior::ir::Location<'ml>,
) -> Operation<'ml> {
    let ty = agg_val.r#type();
    let ty2 = elt.r#type();
    assert!(ty.is_tuple());
    OperationBuilder::new("poison.insert_value", location)
        .add_operands(&[agg_val, elt])
        .add_attributes(&[
            (
                Identifier::new(mlir_ctx, "index"),
                IntegerAttribute::new(Type::from(IntegerType::new(mlir_ctx, 64)), idx as i64)
                    .into(),
            ),
            (
                Identifier::new(mlir_ctx, "to_remove"),
                StringAttribute::new(mlir_ctx, "poison").into(),
            ),
        ])
        .add_results(&[ty])
        .build()
        .expect("valid operation")
}

// See insert_value.
// See immediate_or_packed_pair at rustc_codegen_ssa/src/mir/operand.rs
pub fn decode_ret_value<'ml: 'a, 'a>(
    _mlir_ctx: &'ml crate::mlir::Context,
    val: Value<'ml, 'a>,
    results: &mut Vec<Option<Value<'ml, 'a>>>,
    location: melior::ir::Location<'ml>,
) -> Result<(), crate::error::GpuCodegenError> {
    let ty = val.r#type();
    assert!(ty.is_tuple());
    debug!("decode_ret_value: val type: {ty:?} val:{val:?} at location {location:?}");
    let opval =
        OperationResult::<'ml, 'a>::try_from(val).expect("expected a tuple value for insert_value");
    let op: &Operation<'ml> = unsafe { opval.owner().to_ref() };
    assert!(op.result_count() == 1);
    let ty = op.result(0).unwrap().r#type();
    assert!(ty.is_tuple());
    let tuple = TupleType::<'ml>::try_from(ty).unwrap();
    if results.len() < tuple.type_count() {
        for i in results.len()..tuple.type_count() {
            results.push(None);
        }
    } else if results.len() > tuple.type_count() {
        return Err(crate::error::GpuCodegenError::InvalidMLIR(
            "poison.insert_value should not produce more values than the tuple type".to_string(),
        ));
    }
    match op.name().as_string_ref().as_str().unwrap() {
        "poison.const" => {
            if results.iter().any(|v| v.is_none()) {
                Err(crate::error::GpuCodegenError::InvalidMLIR(format!(
                    "poison.const should not be used directly {} {:?}",
                    val, results
                )))
            } else {
                Ok(())
            }
        }
        "poison.insert_value" => {
            let agg_val: Value<'ml, 'a> = op.operand(0).unwrap();
            let elt: Value<'ml, 'a> = op.operand(1).unwrap();
            let idx: usize = IntegerAttribute::try_from(op.attribute("index").unwrap())
                .unwrap()
                .value() as usize;
            if results[idx].is_some() {
                return Ok(());
            }
            results[idx] = Some(elt);
            decode_ret_value(_mlir_ctx, agg_val, results, location)
        }
        _ => Ok(()),
    }
}
