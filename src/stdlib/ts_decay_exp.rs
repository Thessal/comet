use crate::{RingBufferF64, UnaryOp, export_unary};

#[repr(C)]
pub struct TsDecayExpState {
    pub ema: RingBufferF64,
    pub period: usize,
}

impl UnaryOp for TsDecayExpState {
    fn new(period: usize, len: usize) -> Self {
        TsDecayExpState {
            ema: RingBufferF64::new(1, len),
            period,
        }
    }

    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        
        let alpha = 2.0 / (self.period as f64 + 1.0);
        
        if self.ema.count == 0 {
            // First step
            self.ema.push(a_slice);
            for i in 0..len {
                out_slice[i] = a_slice[i];
            }
            return;
        }
        
        let ema_mut = self.ema.get_latest_mut().unwrap();
        
        for i in 0..len {
            let val = a_slice[i];
            let old_ema = ema_mut[i];
            
            let new_ema = if val.is_nan() {
                old_ema // Skip missing value and carry forward
            } else if old_ema.is_nan() {
                val // Initialize if previous was NaN
            } else {
                val * alpha + old_ema * (1.0 - alpha)
            };
            
            ema_mut[i] = new_ema;
            out_slice[i] = new_ema;
        }
    }
    
    fn drop_buffers(&mut self) {
        self.ema.drop_inner();
    }
}
export_unary!(TsDecayExpState, ts_decay_exp);
