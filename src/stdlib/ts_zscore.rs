use crate::{RingBufferF64, UnaryOp};

#[repr(C)]
pub struct TsZscoreState {
    pub history: RingBufferF64,
}

impl UnaryOp for TsZscoreState {
    fn new(period: usize, len: usize) -> Self {
        TsZscoreState {
            history: RingBufferF64::new(period, len),
        }
    }

    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        
        self.history.push(a_slice);

        let period = self.history.cap;

        if self.history.count < period {
            out_slice.fill(f64::NAN);
            return;
        }

        let head = self.history.head;
        let ptr = self.history.ptr;

        for i in 0..len {
            let mut sum = 0.0;
            let mut count = 0;
            for j in 0..period {
                let row_idx = (head + j) % period;
                let slice = unsafe { std::slice::from_raw_parts(ptr.add(row_idx * len), len) };
                let val = slice[i];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }

            if count < 2 {
                // Std is 0 for 1 element. Z-score requires division by std. If std is 0, z-score is NaN.
                out_slice[i] = f64::NAN;
            } else {
                let mean = sum / (count as f64);
                let mut var_sum = 0.0;
                for j in 0..period {
                    let row_idx = (head + j) % period;
                    let slice = unsafe { std::slice::from_raw_parts(ptr.add(row_idx * len), len) };
                    let val = slice[i];
                    if !val.is_nan() {
                        let diff = val - mean;
                        var_sum += diff * diff;
                    }
                }
                let std = (var_sum / (count as f64)).sqrt();
                
                let cur_val = a_slice[i];
                if cur_val.is_nan() || std == 0.0 {
                    out_slice[i] = f64::NAN;
                } else {
                    out_slice[i] = (cur_val - mean) / std;
                }
            }
        }
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
    }
}
