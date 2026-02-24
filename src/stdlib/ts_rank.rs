use crate::{RingBufferF64, UnaryOp, export_unary};

#[repr(C)]
pub struct TsRankState {
    pub buffers: Vec<Vec<f64>>,
    pub history: RingBufferF64,
    pub time: usize,
    pub period: usize,
}

impl UnaryOp for TsRankState {
    fn new(period: usize, len: usize) -> Self {
        TsRankState {
            buffers: vec![Vec::with_capacity(period); len],
            history: RingBufferF64::new(period, len),
            time: 0,
            period,
        }
    }
    
    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { std::slice::from_raw_parts(a_ptr, len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        let old_history_slice_opt = self.history.get_oldest();
        
        for j in 0..len {
            let v = a_slice[j];
            
            // Remove leaving value
            if self.time >= self.period {
                if let Some(old_history_slice) = old_history_slice_opt {
                    let old_v = old_history_slice[j];
                    if !old_v.is_nan() {
                        if let Ok(idx) = self.buffers[j].binary_search_by(|a| a.partial_cmp(&old_v).unwrap_or(std::cmp::Ordering::Equal)) {
                            // Find any element and carefully iterate to remove just one instance
                            // we just remove the first one found since all identical values are equivalent
                            self.buffers[j].remove(idx);
                        }
                    }
                }
            }
            
            // Insert new value
            if !v.is_nan() {
                let idx = match self.buffers[j].binary_search_by(|a| a.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Equal)) {
                    Ok(i) => i,
                    Err(i) => i,
                };
                self.buffers[j].insert(idx, v);
            }
            
            // Compute rank
            if !v.is_nan() && self.buffers[j].len() > 1 {
                let mut left = 0;
                let mut right = 0;
                for (i, &val) in self.buffers[j].iter().enumerate() {
                    if val < v {
                        left = i + 1;
                    }
                    if val <= v {
                        right = i + 1;
                    }
                }
                let avg_rank = left as f64 + (right as f64 - left as f64 + 1.0) / 2.0;
                out_slice[j] = (avg_rank - 1.0) / ((self.buffers[j].len() - 1) as f64);
            } else {
                out_slice[j] = f64::NAN;
            }
        }
        
        self.history.push(a_slice);
        self.time += 1;
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
        self.buffers.clear();
    }
}
export_unary!(TsRankState, ts_rank);
