use crate::{CometData, DequeState, MatrixOp, PartialDeque};

#[repr(C)]
pub struct CovarianceState {
    pub history: DequeState,
    pub period: usize,
    pub time: usize,
    pub len: usize,
}

impl CovarianceState {
    pub fn new(period: usize, len: usize) -> Self {
        CovarianceState {
            history: DequeState::new(period, len),
            period,
            time: 0,
            len,
        }
    }
}

// Calculates Sample Covariance Matrix
impl MatrixOp for CovarianceState {
    fn step(&mut self, returns: CometData, out_ptr: *mut f64) {
        let len = self.len;
        let returns_slice = unsafe { returns.as_slice(len) };
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len * len) };

        self.history.push(returns_slice);
        self.time += 1;

        if self.time < self.period {
            for i in 0..(len * len) {
                out[i] = f64::NAN;
            }
            return;
        }

        let mut means = vec![0.0; len];
        let mut counts = vec![0.0; len];

        for row in self.history.history.iter() {
            for i in 0..len {
                if !row[i].is_nan() {
                    means[i] += row[i];
                    counts[i] += 1.0;
                }
            }
        }

        for i in 0..len {
            if counts[i] > 0.0 {
                means[i] /= counts[i];
            } else {
                means[i] = f64::NAN;
            }
        }

        for i in 0..len {
            for j in 0..len {
                let out_idx = i * len + j;
                if means[i].is_nan() || means[j].is_nan() {
                    out[out_idx] = f64::NAN;
                    continue;
                }

                let mut cov = 0.0;
                let mut pair_count = 0.0;

                for row in self.history.history.iter() {
                    if !row[i].is_nan() && !row[j].is_nan() {
                        cov += (row[i] - means[i]) * (row[j] - means[j]);
                        pair_count += 1.0;
                    }
                }

                if pair_count > 1.0 {
                    out[out_idx] = cov / (pair_count - 1.0);
                } else {
                    out[out_idx] = f64::NAN;
                }
            }
        }
    }
}

inventory::submit! {
    crate::OperatorMeta {
        name: "covariance",
        output_shape: crate::OutputShape::Matrix,
    }
}
