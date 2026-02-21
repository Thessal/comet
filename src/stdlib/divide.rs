use crate::{BinaryOp, export_binary};
use std::slice;

#[repr(C)]
pub struct DivideState {
}

impl BinaryOp for DivideState {
    fn new(_period: usize, _len: usize) -> Self {
        DivideState {}
    }
    
    fn step(&mut self, a_ptr: *const f64, b_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a = unsafe { slice::from_raw_parts(a_ptr, len) };
        let b = unsafe { slice::from_raw_parts(b_ptr, len) };
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, len) };

        for i in 0..len {
            out[i] = a[i] / b[i];
        }
    }
}

export_binary!(DivideState, divide);
