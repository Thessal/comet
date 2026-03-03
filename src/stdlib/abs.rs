use crate::{UnaryOp};
use std::slice;
#[repr(C)]
pub struct AbsState {
}

impl UnaryOp for AbsState {
    fn new(_period: usize, _len: usize) -> Self {
        AbsState {}
    }
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64, len: usize) {
        let a_ptr = unsafe { a.as_slice(len).as_ptr() };
        let a = unsafe { slice::from_raw_parts(a_ptr, len) };
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, len) };

        for i in 0..len {
            out[i] = a[i].abs();
        }
    }
}
