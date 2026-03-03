use crate::{UnaryOp};

#[repr(C)]
pub struct TsFfillState {
    pub last_valid: Vec<f64>,
    pub dist: Vec<usize>,
    pub period: usize,
    pub len: usize,
}

impl TsFfillState {
    pub fn new(period: usize, len: usize) -> Self {
        TsFfillState {
            last_valid: vec![f64::NAN; len],
            dist: vec![0; len],
            period, len }
    }
}
impl UnaryOp for TsFfillState {

    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_slice = unsafe { a.as_slice(len) };
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


inventory::submit! {
    crate::OperatorMeta {
        name: "ts_ffill",
        output_shape: crate::OutputShape::DataFrame,
    }
}
