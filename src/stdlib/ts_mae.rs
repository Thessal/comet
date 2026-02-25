use crate::{RingBufferF64, UnaryOp, export_unary};

#[repr(C)]
pub struct TsMaeState {
    pub history: RingBufferF64,
}

impl UnaryOp for TsMaeState {
    fn new(period: usize, len: usize) -> Self {
        TsMaeState {
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
            } else {
                let mean = sum / (count as f64);
                let mut err_sum = 0.0;
                for j in 0..period {
                    let row_idx = (head + j) % period;
                    let slice = unsafe { std::slice::from_raw_parts(ptr.add(row_idx * len), len) };
                    let val = slice[i];
                    if !val.is_nan() {
                        err_sum += (val - mean).abs();
                    }
                }
                out_slice[i] = err_sum / (count as f64);
            }
        }
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
    }
}
export_unary!(TsMaeState, ts_mae);
