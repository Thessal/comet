use crate::{BinaryOp, CometData, DataType};

#[repr(C)]
pub struct MultiplyState {
    pub len: usize,
}

impl MultiplyState {
    pub fn new(_period: usize, len: usize) -> Self {
        MultiplyState { len }
    }
}
impl BinaryOp for MultiplyState {
    
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64) {
        let len = self.len;
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        match (a.dtype, b.dtype) {
            (DataType::DataFrame, DataType::DataFrame) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = a_sl[i] * b_sl[i];
                }
            }
            (DataType::DataFrame, _) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_val = unsafe { b.get_scalar() };
                for i in 0..len {
                    out[i] = a_sl[i] * b_val;
                }
            }
            (_, DataType::DataFrame) => {
                let a_val = unsafe { a.get_scalar() };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = a_val * b_sl[i];
                }
            }
             _ => {
                out[0] = unsafe { a.get_scalar() * b.get_scalar() };
            }
        }
    }
}


inventory::submit! {
    crate::OperatorMeta {
        name: "multiply",
        output_shape: crate::OutputShape::DataFrame,
    }
}
