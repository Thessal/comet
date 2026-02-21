// src/stdlib/cs_zscore.rs
use crate::{UnaryOp, export_unary};

#[repr(C)]
pub struct CsZscoreState;

impl UnaryOp for CsZscoreState {
    fn new(_period: usize, _len: usize) -> Self {
        CsZscoreState
    }

    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
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

export_unary!(CsZscoreState, cs_zscore);
