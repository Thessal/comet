use crate::{PartialDeque, DequeState, UnaryOp};

#[repr(C)]
pub struct TsDecayLinearState {
    pub history: DequeState,
}

impl UnaryOp for TsDecayLinearState {
    fn new(period: usize, len: usize) -> Self {
        TsDecayLinearState {
            history: DequeState::new(period, len),
        }
    }

    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        self.history.push(a_slice);

        let period = self.history.cap;
        let sum_weights = (period as f64 * (period as f64 + 1.0)) / 2.0;

        if self.history.count < period {
            out_slice.fill(f64::NAN);
            return;
        }

        for i in 0..len {
            let mut sum = 0.0;
            for (j, row) in self.history.history.iter().enumerate() {
                let weight = (j + 1) as f64;
                sum += row[i] * weight;
                // Note: if row[i] is NaN, sum becomes NaN natively
            }
            out_slice[i] = sum / sum_weights;
        }
    }

    fn drop_buffers(&mut self) {}
}
