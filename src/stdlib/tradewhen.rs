use crate::{CometData, DataType, TernaryOp};

#[derive(Clone)]
pub struct TradewhenState {
    pub last_val: Vec<f64>,
    pub time_since: Vec<usize>,
    pub period: usize,
}

impl TernaryOp for TradewhenState {
    fn new(period: usize, len: usize) -> Self {
        TradewhenState {
            last_val: vec![f64::NAN; len],
            time_since: vec![usize::MAX; len],
            period,
        }
    }

    fn step(&mut self, signal: CometData, enter: CometData, exit: CometData, out_ptr: *mut f64, len: usize) {
        let is_sig_df = signal.dtype == DataType::DataFrame;
        let is_ent_df = enter.dtype == DataType::DataFrame;
        let is_ext_df = exit.dtype == DataType::DataFrame;

        let sig_scalar = if !is_sig_df { unsafe { signal.get_scalar() } } else { f64::NAN };
        let ent_scalar = if !is_ent_df { unsafe { enter.get_scalar() } } else { 0.0 };
        let ext_scalar = if !is_ext_df { unsafe { exit.get_scalar() } } else { 0.0 };

        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        for i in 0..len {
            let sig = if is_sig_df { unsafe { signal.as_slice(len)[i] } } else { sig_scalar };
            let ent = if is_ent_df { unsafe { enter.as_slice(len)[i] } } else { ent_scalar };
            let ext = if is_ext_df { unsafe { exit.as_slice(len)[i] } } else { ext_scalar };

            let enter_cond = ent > 0.0 && !ent.is_nan();
            let exit_cond = ext > 0.0 && !ext.is_nan() && !enter_cond;

            let mut input_val = std::f64::NAN;
            if enter_cond {
                input_val = sig;
            } else if exit_cond {
                input_val = std::f64::INFINITY;
            }

            // Apply ffill logic on input_val
            if !input_val.is_nan() {
                self.last_val[i] = input_val;
                self.time_since[i] = 0;
            } else {
                if self.time_since[i] != usize::MAX {
                    self.time_since[i] = self.time_since[i].saturating_add(1);
                }
            }

            if self.time_since[i] <= self.period {
                if self.last_val[i].is_infinite() {
                    out[i] = f64::NAN;
                } else {
                    out[i] = self.last_val[i];
                }
            } else {
                out[i] = f64::NAN;
            }
        }
    }
}
