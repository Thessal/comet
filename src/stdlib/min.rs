use crate::{BinaryOp, CometData, DataType, export_binary};

#[repr(C)]
pub struct MinState {}

impl BinaryOp for MinState {
    fn new(_period: usize, _len: usize) -> Self {
        MinState {}
    }
    
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64, len: usize) {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        match (a.dtype, b.dtype) {
            (DataType::DataFrame, DataType::DataFrame) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = if a_sl[i].is_nan() || b_sl[i].is_nan() { f64::NAN } else { a_sl[i].min(b_sl[i]) };
                }
            }
            (DataType::DataFrame, _) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_val = unsafe { b.get_scalar() };
                for i in 0..len {
                    out[i] = if a_sl[i].is_nan() || b_val.is_nan() { f64::NAN } else { a_sl[i].min(b_val) };
                }
            }
            (_, DataType::DataFrame) => {
                let a_val = unsafe { a.get_scalar() };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = if a_val.is_nan() || b_sl[i].is_nan() { f64::NAN } else { a_val.min(b_sl[i]) };
                }
            }
             _ => {
                let a_val = unsafe { a.get_scalar() };
                let b_val = unsafe { b.get_scalar() };
                out[0] = if a_val.is_nan() || b_val.is_nan() { f64::NAN } else { a_val.min(b_val) };
            }
        }
    }
}

export_binary!(MinState, min);
