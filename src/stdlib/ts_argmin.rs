use crate::{PartialDeque, DequeState, UnaryOp};

#[repr(C)]
pub struct TsArgminState {
    pub history: DequeState,
    pub len: usize,
}

impl TsArgminState {
    pub fn new(period: usize, len: usize) -> Self {
        TsArgminState {
            history: DequeState::new(period, len), len }
    }
}
impl UnaryOp for TsArgminState {

    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        self.history.push(a_slice);

        let period = self.history.cap;

        if self.history.count < period {
            out_slice.fill(f64::NAN);
            return;
        }

        for i in 0..len {
            let mut min_val = f64::INFINITY;
            let mut min_idx = -1_isize;

            for (j, row) in self.history.history.iter().enumerate() {
                let val = row[i];
                if !val.is_nan() && val < min_val {
                    min_val = val;
                    min_idx = j as isize;
                }
            }

            if min_idx >= 0 {
                out_slice[i] = (min_idx as f64) / (period as f64);
            } else {
                out_slice[i] = f64::NAN;
            }
        }
    }

    fn drop_buffers(&mut self) {}
}


inventory::submit! {
    crate::OperatorMeta {
        name: "ts_argmin",
        output_shape: crate::OutputShape::DataFrame,
    }
}
