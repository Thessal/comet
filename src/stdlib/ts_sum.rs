use crate::{DequeState, PartialDeque, UnaryOp};

#[repr(C)]
pub struct TsSumState {
    pub sum: DequeState,
    pub count: DequeState,
    pub history: DequeState,
    pub len: usize,
}

impl TsSumState {
    pub fn new(period: usize, len: usize) -> Self {
        TsSumState {
            sum: DequeState::new(1, len),
            count: DequeState::new(1, len),
            history: DequeState::new(period, len), len }
    }
}
impl UnaryOp for TsSumState {
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        
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
            
            if let Some(old_history_slice) = old_history_slice_opt {
                let old_val = old_history_slice[i];
                if !old_val.is_nan() {
                    sum_slice[i] -= old_val;
                    count_slice[i] -= 1.0;
                }
            }

            if !val.is_nan() {
                sum_slice[i] += val;
                count_slice[i] += 1.0;
            }
            
            if count_slice[i] <= 0.0 { 
                out_slice[i] = f64::NAN;
            } else {
                out_slice[i] = sum_slice[i];
            }
        }
        
        self.history.push(a_slice);
    }
    
    fn drop_buffers(&mut self) {
    }
}


inventory::submit! {
    crate::OperatorMeta {
        name: "ts_sum",
        output_shape: crate::OutputShape::DataFrame,
    }
}
