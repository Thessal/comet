// src/stdlib/cs_rank.rs
use crate::{UnaryOp, export_unary};

#[repr(C)]
pub struct CsRankState;

impl UnaryOp for CsRankState {
    fn new(_period: usize, _len: usize) -> Self {
        CsRankState
    }

    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        out_slice.fill(f64::NAN);

        let mut valid_vals: Vec<(usize, f64)> = a_slice
            .iter()
            .enumerate()
            .filter(|x| !x.1.is_nan())
            .map(|x| (x.0, *x.1))
            .collect();

        if valid_vals.is_empty() {
            return;
        }

        valid_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut j = 0;
        let n = valid_vals.len();
        while j < n {
            let mut k = j;
            while k < n && valid_vals[k].1 == valid_vals[j].1 {
                k += 1;
            }
            let avg_rank = 0.5 * ((j + 1 + k) as f64);
            for idx in j..k {
                out_slice[valid_vals[idx].0] = avg_rank;
            }
            j = k;
        }
    }
}

export_unary!(CsRankState, cs_rank);
