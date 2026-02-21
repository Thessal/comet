use crate::{UnaryOp, export_unary};

#[repr(C)]
pub struct TsFfillState {
    pub last_valid: Vec<f64>,
    pub dist: Vec<usize>,
    pub period: usize,
}

impl UnaryOp for TsFfillState {
    fn new(period: usize, len: usize) -> Self {
        TsFfillState {
            last_valid: vec![f64::NAN; len],
            dist: vec![0; len],
            period,
        }
    }

    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        for i in 0..len {
            let val = a_slice[i];
            if !val.is_nan() {
                self.last_valid[i] = val;
                self.dist[i] = 0;
                out_slice[i] = val;
            } else {
                if !self.last_valid[i].is_nan() && self.dist[i] < self.period {
                    self.dist[i] = self.dist[i].saturating_add(1);
                    out_slice[i] = self.last_valid[i];
                } else {
                    self.last_valid[i] = f64::NAN;
                    self.dist[i] = self.dist[i].saturating_add(1);
                    out_slice[i] = f64::NAN;
                }
            }
        }
    }
    
    fn drop_buffers(&mut self) {
        // Rust Vec drops automatically when TsFfillState drops.
    }
}
export_unary!(TsFfillState, ts_ffill);
