use crate::{RingBufferF64, UnaryOp};
use std::collections::VecDeque;

#[repr(C)]
pub struct TsMinState {
    pub deques: Vec<VecDeque<(usize, f64)>>,
    pub valid_counts: Vec<usize>,
    pub history: RingBufferF64,
    pub time: usize,
    pub period: usize,
}

impl UnaryOp for TsMinState {
    fn new(period: usize, len: usize) -> Self {
        TsMinState {
            deques: vec![VecDeque::new(); len],
            valid_counts: vec![0; len],
            history: RingBufferF64::new(period, len),
            time: 0,
            period,
        }
    }
    
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64, len: usize) {
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        
        let old_history_slice_opt = self.history.get_oldest();
        
        for j in 0..len {
            let v = a_slice[j];
            
            // expire old index
            if let Some(dq) = self.deques.get_mut(j) {
                while let Some(&(t_idx, _)) = dq.front() {
                    if self.time >= self.period && self.time - t_idx >= self.period {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            
            if self.time >= self.period {
                if let Some(old_history_slice) = old_history_slice_opt {
                    if !old_history_slice[j].is_nan() {
                        self.valid_counts[j] -= 1;
                    }
                }
            }
            
            if !v.is_nan() {
                self.valid_counts[j] += 1;
                if let Some(dq) = self.deques.get_mut(j) {
                    while let Some(&(_, last_v)) = dq.back() {
                        if last_v > v {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back((self.time, v));
                }
            }
            
            if self.valid_counts[j] > 0 {
                out_slice[j] = self.deques[j].front().map(|x| x.1).unwrap_or(f64::NAN);
            } else {
                out_slice[j] = f64::NAN;
            }
        }
        
        self.history.push(a_slice);
        self.time += 1;
    }
    
    fn drop_buffers(&mut self) {
        self.history.drop_inner();
        self.deques.clear();
        self.valid_counts.clear();
    }
}
