use crate::{RingBufferF64, UnaryOp, export_unary};

#[repr(C)]
pub struct TsDecayLinearState {
    pub history: RingBufferF64,
}

impl UnaryOp for TsDecayLinearState {
    fn new(period: usize, len: usize) -> Self {
        TsDecayLinearState {
            history: RingBufferF64::new(period, len),
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
            let head = self.history.head;
            let ptr = self.history.ptr;
            // The oldest element is at `head` (since count == cap).
            for j in 0..period {
                let weight = (j + 1) as f64;
                let row_idx = (head + j) % period;
                let slice = unsafe {
                    std::slice::from_raw_parts(ptr.add(row_idx * len), len)
                };
                sum += slice[i] * weight; 
                // Note: if slice[i] is NaN, sum becomes NaN natively
            }
            out_slice[i] = sum / sum_weights;
        }
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
    }
}
export_unary!(TsDecayLinearState, ts_decay_linear);
