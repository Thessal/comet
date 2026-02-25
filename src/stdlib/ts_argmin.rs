use crate::{RingBufferF64, UnaryOp, export_unary};

#[repr(C)]
pub struct TsArgminState {
    pub history: RingBufferF64,
}

impl UnaryOp for TsArgminState {
    fn new(period: usize, len: usize) -> Self {
        TsArgminState {
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
            let mut min_val = f64::INFINITY;
            let mut min_idx = -1_isize;

            for j in 0..period {
                let row_idx = (head + j) % period;
                let slice = unsafe {
                    std::slice::from_raw_parts(ptr.add(row_idx * len), len)
                };
                let val = slice[i];

                if !val.is_nan() && val < min_val {
                    min_val = val;
                    min_idx = j as isize;
                }
            }

            if min_idx >= 0 {
                out_slice[i] = (min_idx as f64) / (period as f64);
            } else {
                out_slice[i] = f64::NAN;
            }
        }
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
    }
}
export_unary!(TsArgminState, ts_argmin);
