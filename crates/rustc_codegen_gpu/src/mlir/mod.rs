pub(crate) mod attr;
pub(crate) mod gpu;
pub(crate) mod memref;
pub(crate) mod poison;
pub(crate) mod visit;

use melior::Context;
use melior::dialect::ods::func::FuncOperation;
use melior::dialect::ods::gpu::{GPUFuncOperation, GPUModuleOperation};
use melior::dialect::ods::memref::GetGlobalOperation;
use melior::dialect::{DialectRegistry, arith, func};
use melior::ir::attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute};
use melior::ir::operation::{OperationLike, OperationMutLike, OperationResult};
use melior::ir::r#type::{FunctionType, IntegerType, MemRefType, RankedTensorType, TupleType};
use melior::ir::*;
use melior::utility::register_all_dialects;
use rustc_span::Loc;

pub(crate) fn create_mlir_ctx() -> &'static melior::Context {
    let ctx = Box::leak(Box::new(melior::Context::new()));
    // We need a registry to hold all the dialects
    let registry = DialectRegistry::new();
    // Register all dialects that come with MLIR.
    register_all_dialects(&registry);

    // The MLIR context, like the LLVM one.
    ctx.append_dialect_registry(&registry);
    ctx.load_all_available_dialects();
    assert!(ctx.is_registered_operation("gpu.thread_id"));

    /*let content = std::fs::read_to_string(PathBuf::from("~/test.mlir")).unwrap();
    let m = melior::ir::Module::parse(
        &ctx,
        content.as_str()
    );
    log::debug!("parsed mlir module: {:?}", m);
    log::debug!("attr: {:?}", m.unwrap().body().first_operation().unwrap().block().unwrap().first_operation().unwrap().attribute("dimension"));*/
    ctx
}

pub(crate) fn new_empty_module<'ml>(ctx: &'ml Context) -> melior::ir::Module<'ml> {
    let location = Location::unknown(ctx);
    Module::new(location)
}

pub(crate) fn create_top_module<'ml>(
    ctx: &'ml Context,
) -> (melior::ir::Module<'ml>, melior::ir::BlockRef<'ml, 'ml>, melior::ir::BlockRef<'ml, 'ml>) {
    let location = Location::unknown(ctx);
    let mut module = Module::new(location);
    let unit_attr = melior::ir::Attribute::unit(ctx);

    module.as_operation_mut().set_attribute("gpu.container_module", unit_attr);
    let region = Region::new();
    let block = Block::new(&[]);
    let gpu_block = region.append_block(block);
    let mut gpu_mod: Operation<'ml> =
        melior::dialect::ods::gpu::module(ctx, region, StringAttribute::new(ctx, "gpu"), location)
            .into();
    gpu_mod.set_attribute("visibility", StringAttribute::new(ctx, "public").into());
    let region = Region::new();
    let block = Block::new(&[]);
    let cpu_block = region.append_block(block);
    let cpu_mod = melior::dialect::ods::builtin::module(ctx, region, location);

    module.body().append_operation(gpu_mod);
    module.body().append_operation(cpu_mod.into());
    (module, gpu_block, cpu_block)
}

pub enum MLIRVisibility {
    Public,
    Private,
    Nested,
}

const VALUE_SYM: &str = "value";
const FUNCTION_TYPE_SYM: &str = "function_type";
const VISIBILITY_SYM: &str = "sym_visibility";
const NAME_SYM: &str = "name";
pub const BUILTIN_SYM: &str = "gpu_codegen_builtin";
pub const SYM_NAME_SYM: &str = "sym_name";

pub trait ValueToOpRef<'ml, 'a> {
    fn to_func_sym(&self) -> Result<FlatSymbolRefAttribute<'ml>, melior::Error>;
    fn to_get_global_name(&self) -> Result<FlatSymbolRefAttribute<'ml>, melior::Error>;
    fn is_from_op(&self, sym: Option<&'static str>) -> Result<Operation<'ml>, melior::Error>;
    fn get_op_attr<T: TryFrom<Attribute<'ml>>>(
        &self,
        attr_name: &'static str,
    ) -> Result<T, melior::Error>;
}

impl<'ml, 'a> ValueToOpRef<'ml, 'a> for Value<'ml, 'a> {
    fn get_op_attr<T: TryFrom<Attribute<'ml>>>(
        &self,
        attr_name: &'static str,
    ) -> Result<T, melior::Error> {
        let op_val: OperationResult<'ml, 'a> = (*self).try_into()?;
        let attr = op_val.owner().attribute(attr_name)?;
        attr.try_into()
            .map_err(|e| melior::Error::AttributeExpected(attr_name, "get_attr_if_sym".to_string()))
    }

    fn to_func_sym(&self) -> Result<FlatSymbolRefAttribute<'ml>, melior::Error> {
        let op_val: OperationResult<'ml, 'a> = (*self).try_into()?;
        let op = op_val.owner();
        op.get_attr_if_sym("func.constant", VALUE_SYM)
    }

    fn to_get_global_name(&self) -> Result<FlatSymbolRefAttribute<'ml>, melior::Error> {
        let op_val: OperationResult<'ml, 'a> = (*self).try_into()?;
        let op = op_val.owner();
        op.get_attr_if_sym(GetGlobalOperation::name(), NAME_SYM)
    }

    fn is_from_op(&self, sym: Option<&'static str>) -> Result<Operation<'ml>, melior::Error> {
        let Ok(op_val) = OperationResult::<'ml, 'a>::try_from(*self) else {
            return Err(melior::Error::ResultNotFound("is_from_op"));
        };
        let op = op_val.owner();
        let Some(sym) = sym else {
            return Ok((*op).clone());
        };
        op.expect_op_by_sym(sym)?;
        Ok((*op).clone())
    }
}

pub trait BlockRefWithTime<'ml, 'a> {
    unsafe fn to_ref(&self) -> &'a Block<'ml>;
}

impl<'ml, 'a> BlockRefWithTime<'ml, 'a> for BlockRef<'ml, 'a> {
    unsafe fn to_ref(&self) -> &'a Block<'ml> {
        unsafe { std::mem::transmute(self) }
    }
}
impl From<MLIRVisibility> for &str {
    fn from(visibility: MLIRVisibility) -> Self {
        match visibility {
            MLIRVisibility::Public => "public",
            MLIRVisibility::Private => "private",
            MLIRVisibility::Nested => "nested",
        }
    }
}

pub trait MLIROpHelpers<'ml> {
    fn get_op_operands_types(&self) -> Vec<melior::ir::r#type::Type<'ml>>;
    fn is_gpu_func(&self) -> bool;
    fn is_kernel_func(&self) -> bool;
    fn expect_op_by_sym(&self, sym: &'static str) -> Result<(), melior::Error>;
    fn get_attr_if_sym<A: TryFrom<melior::ir::Attribute<'ml>>>(
        &self,
        sym: &'static str,
        attr_name: &'static str,
    ) -> Result<A, melior::Error>;
    fn get_func_type(&self) -> Result<FunctionType<'ml>, melior::Error>;
}

pub trait MLIRMutOpHelpers<'ml> {
    fn set_op_visible<'a: 'ml>(&mut self, ctx: &'a Context, val: MLIRVisibility);
}

impl<'ml> MLIRMutOpHelpers<'ml> for Operation<'ml> {
    fn set_op_visible<'a: 'ml>(&mut self, ctx: &'a Context, val: MLIRVisibility) {
        let attr = melior::ir::attribute::StringAttribute::new(ctx, val.into());
        self.set_attribute(VISIBILITY_SYM, attr.into());
    }
}

impl<'ml, 'a> MLIROpHelpers<'ml> for OperationRef<'ml, 'a> {
    fn is_gpu_func(&self) -> bool {
        if self.expect_op_by_sym(GPUFuncOperation::name()).is_ok() {
            return true;
        }
        if let Some(m) = self.parent_operation() {
            m.expect_op_by_sym(GPUModuleOperation::name()).is_ok()
        } else {
            false
        }
    }

    fn is_kernel_func(&self) -> bool {
        self.expect_op_by_sym(GPUFuncOperation::name()).is_ok()
    }

    fn expect_op_by_sym(&self, sym: &'static str) -> Result<(), melior::Error> {
        if self.name().as_string_ref().as_str().unwrap() == sym {
            Ok(())
        } else {
            Err(melior::Error::OperandNotFound(sym))
        }
    }

    fn get_attr_if_sym<A: TryFrom<melior::ir::Attribute<'ml>>>(
        &self,
        sym: &'static str,
        attr_name: &'static str,
    ) -> Result<A, melior::Error> {
        self.expect_op_by_sym(sym)?;
        self.attribute(attr_name)?
            .try_into()
            .map_err(|e| melior::Error::AttributeExpected(attr_name, "get_attr_if_sym".to_string()))
    }

    fn get_func_type(&self) -> Result<FunctionType<'ml>, melior::Error> {
        self.get_attr_if_sym::<TypeAttribute>(GPUFuncOperation::name(), FUNCTION_TYPE_SYM)
            .or(self.get_attr_if_sym::<TypeAttribute>(FuncOperation::name(), FUNCTION_TYPE_SYM))
            .map(|attr| {
                attr.value().try_into().map_err(|e| {
                    melior::Error::TypeExpected(FUNCTION_TYPE_SYM, "get_func_type".to_string())
                })
            })
            .and_then(|func_type| func_type)
    }

    fn get_op_operands_types(&self) -> Vec<melior::ir::r#type::Type<'ml>> {
        let func_type = self.get_func_type().unwrap();
        let mut ret = vec![];
        for i in 0..func_type.input_count() {
            ret.push(func_type.input(i).unwrap());
        }
        ret
    }
}

#[allow(dead_code)]
pub(crate) fn generate_test_module<'a>(loc: Loc) -> Module<'a> {
    // We need a registry to hold all the dialects
    let registry = DialectRegistry::new();
    // Register all dialects that come with MLIR.
    register_all_dialects(&registry);

    // The MLIR context, like the LLVM one.
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // A location is a debug location like in LLVM, in MLIR all
    // operations need a location, even if its "unknown".
    let location = Location::new(
        &context,
        format!("{}", loc.file.name.prefer_local()).as_str(),
        loc.line,
        loc.col.0,
    );

    // A MLIR module is akin to a LLVM module.
    let module = Module::new(location);

    // A integer-like type with platform dependent bit width. (like size_t or usize)
    // This is a type defined in the Builtin dialect.
    let index_type = Type::index(&context);

    // Append a `func::func` operation to the body (a block) of the module.
    // This operation accepts a string attribute, which is the name.
    // A type attribute, which contains a function type in this case.
    // Then it accepts a single region, which is where the body
    // of the function will be, this region can have
    // multiple blocks, which is how you may implement
    // control flow within the function.
    // These blocks each can have more operations.
    module.body().append_operation(func::func(
        &context,
        // accepts a StringAttribute which is the function name.
        StringAttribute::new(&context, "add"),
        // A type attribute, defining the function signature.
        TypeAttribute::new(
            FunctionType::new(&context, &[index_type, index_type], &[index_type]).into(),
        ),
        {
            // The first block within the region, blocks accept arguments
            // In regions with control flow, MLIR leverages
            // this structure to implicitly represent
            // the passage of control-flow dependent values without the complex nuances
            // of PHI nodes in traditional SSA representations.
            let block = Block::new(&[(index_type, location), (index_type, location)]);

            // Use the arith dialect to add the 2 arguments.
            let sum = block.append_operation(arith::addi(
                block.argument(0).unwrap().into(),
                block.argument(1).unwrap().into(),
                location,
            ));

            // Return the result using the "func" dialect return operation.
            block.append_operation(func::r#return(&[sum.result(0).unwrap().into()], location));

            // The Func operation requires a region,
            // we add the block we created to the region and return it,
            // which is passed as an argument to the `func::func` function.
            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    ));

    assert!(module.as_operation().verify());
    module
}

pub(crate) fn mlir_val_to_const_int<'ml, 'a>(value: melior::ir::Value<'ml, 'a>) -> Option<u128> {
    if let Ok(op) = value.is_from_op(Some("arith.constant")) {
        melior::ir::attribute::IntegerAttribute::try_from(op.attribute("value").unwrap())
            .ok()
            .map(|v| v.value() as u128)
    } else {
        None
    }
}

pub(crate) fn float_width(ty: melior::ir::r#type::Type<'_>) -> Option<usize> {
    if ty.is_float() {
        unsafe { Some(mlir_sys::mlirFloatTypeGetWidth(ty.to_raw()) as usize) }
    } else {
        None
    }
}

pub(crate) fn int_width(ty: melior::ir::r#type::Type<'_>) -> Option<u32> {
    if let Ok(t) = IntegerType::try_from(ty) {
        Some(t.width())
    } else if ty.is_index() {
        Some(usize::BITS)
    } else {
        None
    }
}

pub(crate) fn static_size_of(ty: Type<'_>) -> usize {
    if ty.is_integer() {
        let int_ty = IntegerType::try_from(ty).unwrap();
        int_ty.width() as usize / 8
    } else if ty.is_index() || ty.is_mem_ref() {
        size_of::<usize>()
    } else if ty.is_float() {
        float_width(ty).unwrap() / 8
    } else if ty.is_tuple() {
        let tuple = TupleType::try_from(ty).unwrap();
        let mut total_size = 0;
        for i in 0..tuple.type_count() {
            total_size += static_size_of(tuple.r#type(i).unwrap());
        }
        total_size
    } else if ty.is_ranked_tensor() {
        let ranked_ty = RankedTensorType::try_from(ty).unwrap();
        let ty = ranked_ty.element();
        let base_size = static_size_of(ty);
        todo!();
    } else {
        panic!("Unsupported type: {:?}", ty);
    }
}

pub(crate) fn type_memref<'ml>(
    ctx: &'ml Context,
    eletype: Type<'ml>,
    dim: &[i64],
    layout: Option<Attribute<'ml>>,
    memory_space: Option<Attribute<'ml>>,
) -> MemRefType<'ml> {
    if eletype.is_tuple() {
        let size = crate::mlir::static_size_of(eletype);
        assert!(!dim.is_empty());
        let mut dim = dim.to_vec();
        dim.push(size as i64);
        return type_memref(ctx, IntegerType::new(ctx, 8).into(), &dim, layout, memory_space);
    }
    MemRefType::new(eletype, dim, layout, memory_space)
}

pub(crate) fn same_value(idx1: Value<'_, '_>, idx2: Value<'_, '_>) -> bool {
    if idx1 == idx2 {
        return true;
    }
    let v1 = crate::mlir::mlir_val_to_const_int(idx1);
    let v2 = crate::mlir::mlir_val_to_const_int(idx2);
    if v1.is_some() && v1 == v2 {
        return true;
    }
    false
}

pub(crate) fn value_loc<'ml>(val: Value<'ml, '_>) -> Option<Location<'ml>> {
    let op_val: OperationResult<'_, '_> = val.try_into().ok()?;
    Some(op_val.owner().location())
}

pub(crate) fn value_loc_decoded<'ml>(val: Value<'ml, '_>) -> Option<(String, usize, usize)> {
    value_loc(val).and_then(|loc| loc_decoded(loc))
}

pub(crate) fn loc_decoded<'ml>(loc: Location<'ml>) -> Option<(String, usize, usize)> {
    let s = format!("{}", loc);
    let parts = s.split(":").collect::<Vec<_>>();
    if parts.len() != 3 {
        return None;
    }
    Some((parts[0].to_string(), parts[1].parse().unwrap_or(0), parts[2].parse().unwrap_or(0)))
}
