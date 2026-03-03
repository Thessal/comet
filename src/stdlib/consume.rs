use crate::{UnaryOp};

#[repr(C)]
pub struct ConsumeFloatState {}

impl UnaryOp for ConsumeFloatState {
    fn new(_period: usize, _len: usize) -> Self {
        ConsumeFloatState {}
    }
    
    fn step(&mut self, _a: crate::CometData, _out_ptr: *mut f64, _len: usize) {
        // Void function, no output
    }
}
