use crate::UnaryOp;

#[repr(C)]
pub struct ConsumeFloatState {
    pub len: usize,
}

impl ConsumeFloatState {
    pub fn new(_period: usize, len: usize) -> Self {
        ConsumeFloatState { len }
    }
}
impl UnaryOp for ConsumeFloatState {
    fn step(&mut self, _a: crate::CometData, _out_ptr: *mut f64) {
        // let _len = self.len;
        // Void function, no output
    }
}

inventory::submit! {
    crate::OperatorMeta {
        name: "consume",
        output_shape: crate::OutputShape::DataFrame,
    }
}
