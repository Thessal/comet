use crate::UnaryOp;
use std::slice;

#[repr(C)]
pub struct CsRankNonzeroState {
    pub eps: f64,
    pub len: usize,
}

impl CsRankNonzeroState {
    pub fn new(eps: f64, len: usize) -> Self {
        CsRankNonzeroState { eps: eps, len }
    }
}
impl UnaryOp for CsRankNonzeroState {
    fn step(&mut self, signal: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let eps = self.eps;

        let mut csrank_state = crate::cs_rank::CsRankState { len };
        csrank_state.step(signal, out_ptr);

        let out = unsafe { slice::from_raw_parts_mut(out_ptr, len) };
        match signal.dtype {
            crate::DataType::DataFrame => {
                for i in 0..len {
                    out[i] += eps;
                }
            }
            crate::DataType::TimeSeries => {
                out[0] += eps;
            }
            crate::DataType::Constant => {
                out[0] += eps;
            }
        }
    }
}

inventory::submit! {
    crate::OperatorMeta {
        name: "cs_rank_nonzero",
        output_shape: crate::OutputShape::DataFrame,
    }
}
