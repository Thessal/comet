use crate::{RingBufferF64, UnaryOp};

#[repr(C)]
pub struct TsStdState {
    pub history: RingBufferF64,
}

impl UnaryOp for TsStdState {
    fn new(period: usize, len: usize) -> Self {
        TsStdState {
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

            if count == 0 {
                out_slice[i] = f64::NAN;
            } else if count == 1 {
                // np.nanstd of 1 element is 0.0 (ddof=0)
                out_slice[i] = 0.0;
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
                out_slice[i] = (var_sum / (count as f64)).sqrt();
            }
        }
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
    }
}
