use crate::{BinaryOp, CometData, DataType};

#[repr(C)]
pub struct LessState {
    pub len: usize,
}

impl LessState {
    pub fn new(_period: usize, len: usize) -> Self {
        LessState { len }
    }
}
impl BinaryOp for LessState {
    
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64) {
        let len = self.len;
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        match (a.dtype, b.dtype) {
            (DataType::DataFrame, DataType::DataFrame) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = if a_sl[i] < b_sl[i] { 1.0 } else { 0.0 };
                }
            }
            (DataType::DataFrame, _) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_val = unsafe { b.get_scalar() };
                for i in 0..len {
                    out[i] = if a_sl[i] < b_val { 1.0 } else { 0.0 };
                }
            }
            (_, DataType::DataFrame) => {
                let a_val = unsafe { a.get_scalar() };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = if a_val < b_sl[i] { 1.0 } else { 0.0 };
                }
            }
             _ => {
                out[0] = if unsafe { a.get_scalar() < b.get_scalar() } { 1.0 } else { 0.0 };
            }
        }
    }
}


inventory::submit! {
    crate::OperatorMeta {
        name: "less",
        output_shape: crate::OutputShape::DataFrame,
    }
}
