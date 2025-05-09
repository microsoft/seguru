use melior::{
    dialect::{arith, func, DialectRegistry},
    ir::{
        attribute::{StringAttribute, TypeAttribute},
        r#type::FunctionType,
        *,
    },
    utility::register_all_dialects,
    Context,
};
use rustc_span::Loc;

pub enum MLIRVisibility {
    Public,
    Private,
    Nested,
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
    fn set_op_visible<'a: 'ml>(&mut self, ctx: &'a Context, val: MLIRVisibility);
}

impl<'ml> MLIROpHelpers<'ml> for Operation<'ml> {
    fn set_op_visible<'a: 'ml>(&mut self, ctx: &'a Context, val: MLIRVisibility) {
        let attr = melior::ir::attribute::StringAttribute::new(ctx, val.into());
        self.set_attribute("visibility", attr.into());
    }
    fn get_op_operands_types(&self) -> Vec<melior::ir::r#type::Type<'ml>> {
        let type_attr: melior::ir::attribute::TypeAttribute = self
            .attribute("function_type")
            .unwrap()
            .try_into()
            .expect("invalid func attr");
        let func_type: FunctionType = type_attr.value().try_into().expect("invalid function type");
        let mut ret = vec![];
        for i in 0..func_type.input_count() {
            ret.push(func_type.input(i).unwrap());
        }
        ret
    }
}

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
