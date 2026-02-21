use crate::{BinaryOp, CometData, DataType, export_binary};

#[repr(C)]
pub struct DivideState {}

impl BinaryOp for DivideState {
    fn new(_period: usize, _len: usize) -> Self {
        DivideState {}
    }
    
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64, len: usize) {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        match (a.dtype, b.dtype) {
            (DataType::DataFrame, DataType::DataFrame) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = a_sl[i] / b_sl[i];
                }
            }
            (DataType::DataFrame, _) => {
                let a_sl = unsafe { a.as_slice(len) };
                let b_val = unsafe { b.get_scalar() };
                for i in 0..len {
                    out[i] = a_sl[i] / b_val;
                }
            }
            (_, DataType::DataFrame) => {
                let a_val = unsafe { a.get_scalar() };
                let b_sl = unsafe { b.as_slice(len) };
                for i in 0..len {
                    out[i] = a_val / b_sl[i];
                }
            }
            // TS / TS, TS / Const, Const / Const
             _ => {
                out[0] = unsafe { a.get_scalar() / b.get_scalar() };
            }
        }
    }
}

export_binary!(DivideState, divide);
