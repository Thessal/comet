use crate::{UnaryOp};
use std::slice;
#[repr(C)]
pub struct AbsState {
    pub len: usize,
}

impl AbsState {
    pub fn new(_period: usize, len: usize) -> Self {
        AbsState { len }
    }
}
impl UnaryOp for AbsState {
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_ptr = unsafe { a.as_slice(len).as_ptr() };
        let a = unsafe { slice::from_raw_parts(a_ptr, len) };
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, len) };

        for i in 0..len {
            out[i] = a[i].abs();
        }
    }
}


inventory::submit! {
    crate::OperatorMeta {
        name: "abs",
        output_shape: crate::OutputShape::DataFrame,
    }
}
