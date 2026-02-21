// In src/stdlib/ts_mean.rs
use crate::{RingBufferF64, UnaryOp, export_unary};
#[repr(C)]
pub struct TsMeanState {
    pub sum: RingBufferF64,
    pub count: RingBufferF64,
    pub history: RingBufferF64,
}

// 1. Implement the generic trait
impl UnaryOp for TsMeanState {
    fn new(period: usize, len: usize) -> Self {
        TsMeanState {
            sum: RingBufferF64::new(1, len),
            count: RingBufferF64::new(1, len),
            history: RingBufferF64::new(period, len),
        }
    }
    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        
        // 1. Get the oldest slice before overwriting it
        let old_history_slice_opt = self.history.get_oldest();
        
        if self.sum.count == 0 {
            let zeros = vec![0.0; len];
            self.sum.push(&zeros);
            self.count.push(&zeros);
        }
        
        let sum_slice = self.sum.get_latest_mut().unwrap(); 
        let count_slice = self.count.get_latest_mut().unwrap();
        
        for i in 0..len {
            let val = a_slice[i];
            
            // 2. Erase the oldest value if we reached capacity and it wasn't NaN
            if let Some(old_history_slice) = old_history_slice_opt {
                let old_val = old_history_slice[i];
                if !old_val.is_nan() {
                    sum_slice[i] -= old_val;
                    count_slice[i] -= 1.0;
                }
            }

            // 3. Accumulate the incoming value if it isn't NaN
            if !val.is_nan() {
                sum_slice[i] += val;
                count_slice[i] += 1.0;
            }
            
            // 4. Calculate the rolling mean, outputting NaN if we have no valid counts
            if count_slice[i] <= 0.0 { 
                out_slice[i] = f64::NAN;
            } else {
                out_slice[i] = sum_slice[i] / count_slice[i];
            }
        }
        
        // 5. Finally, push the new slice into history memory, overwriting the oldest value.
        self.history.push(a_slice);
    }
    
    fn drop_buffers(&mut self) {
        self.sum.drop_inner();
        self.count.drop_inner();
        self.history.drop_inner();
    }
}
export_unary!(TsMeanState, ts_mean);