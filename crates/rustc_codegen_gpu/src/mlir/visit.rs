use melior::ir::operation::{OperationLike, OperationRefMut};
use melior::ir::{BlockLike, RegionLike};

pub fn visit_ops_recursively<F: Fn(&mut OperationRefMut)>(op: &mut OperationRefMut, f: &F) {
    f(op);
    if let Some(mut op) = op.next_in_block_mut() {
        visit_ops_recursively(&mut op, f);
    }

    for region in op.regions() {
        let mut block = region.first_block();
        while let Some(current_block) = block {
            if let Some(mut inner_op) = current_block.first_operation_mut() {
                visit_ops_recursively(&mut inner_op, f);
            }
            block = current_block.next_in_region();
        }
    }
}
