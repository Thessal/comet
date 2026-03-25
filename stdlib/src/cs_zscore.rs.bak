// src/stdlib/cs_zscore.rs
use crate::UnaryOp;

#[repr(C)]
pub struct CsZscoreState {
    pub len: usize,
}

impl CsZscoreState {
    pub fn new(_period: usize, len: usize) -> Self {
        CsZscoreState { len }
    }
}
impl UnaryOp for CsZscoreState {
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        out_slice.fill(f64::NAN);

        let mut sum = 0.0;
        let mut count = 0;

        for &v in a_slice {
            if !v.is_nan() {
                sum += v;
                count += 1;
            }
        }

        if count <= 1 {
            return;
        }

        let mean = sum / (count as f64);
        let mut variance_sum = 0.0;

        for &v in a_slice {
            if !v.is_nan() {
                let diff = v - mean;
                variance_sum += diff * diff;
            }
        }

        // Polars uses sample standard deviation (N - 1) by default
        let std = (variance_sum / ((count - 1) as f64)).sqrt();

        if std > 0.0 {
            for i in 0..len {
                if !a_slice[i].is_nan() {
                    out_slice[i] = (a_slice[i] - mean) / std;
                }
            }
        }
    }
}


inventory::submit! {
    crate::OperatorMeta {
        name: "cs_zscore",
        output_shape: crate::OutputShape::DataFrame,
    }
}
