use crate::{PartialDeque, DequeState, UnaryOp};

#[repr(C)]
pub struct TsMaeState {
    pub history: DequeState,
    pub len: usize,
}

impl TsMaeState {
    pub fn new(period: usize, len: usize) -> Self {
        TsMaeState {
            history: DequeState::new(period, len), len }
    }
}
impl UnaryOp for TsMaeState {

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
            let mut sum = 0.0;
            let mut count = 0;
            for row in self.history.history.iter() {
                let val = row[i];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }

            if count == 0 {
                out_slice[i] = f64::NAN;
            } else {
                let mean = sum / (count as f64);
                let mut err_sum = 0.0;
                for row in self.history.history.iter() {
                    let val = row[i];
                    if !val.is_nan() {
                        err_sum += (val - mean).abs();
                    }
                }
                out_slice[i] = err_sum / (count as f64);
            }
        }
    }

    fn drop_buffers(&mut self) {}
}


inventory::submit! {
    crate::OperatorMeta {
        name: "ts_mae",
        output_shape: crate::OutputShape::DataFrame,
    }
}
