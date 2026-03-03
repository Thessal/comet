// src/stdlib/lib.rs

pub mod abs;
pub mod add;
pub mod clip;
pub mod r#const;
pub mod consume;
pub mod covariance;
pub mod cs_rank;
pub mod cs_rank_nonzero;
pub mod cs_zscore;
pub mod data;
pub mod divide;
pub mod equals;
pub mod greater;
pub mod less;
pub mod max;
pub mod mid;
pub mod min;
pub mod multiply;
pub mod power;
pub mod subtract;
pub mod tail_to_nan;
pub mod tradewhen;
pub mod ts_argmax;
pub mod ts_argmin;
pub mod ts_argminmax;
pub mod ts_decay_exp;
pub mod ts_decay_linear;
pub mod ts_delay;
pub mod ts_diff;
pub mod ts_ffill;
pub mod ts_mae;
pub mod ts_max;
pub mod ts_mean;
pub mod ts_min;
pub mod ts_rank;
pub mod ts_std;
pub mod ts_sum;
pub mod ts_zscore;
pub mod r#where;

#[cfg(test)]
mod test_abs;
#[cfg(test)]
mod test_clip;
#[cfg(test)]
mod test_const;
#[cfg(test)]
mod test_covariance;
#[cfg(test)]
mod test_cs_rank;
#[cfg(test)]
mod test_cs_zscore;
#[cfg(test)]
mod test_divide;
#[cfg(test)]
mod test_equals;
#[cfg(test)]
mod test_greater;
#[cfg(test)]
mod test_less;
#[cfg(test)]
mod test_max;
#[cfg(test)]
mod test_mid;
#[cfg(test)]
mod test_min;
#[cfg(test)]
mod test_multiply;
#[cfg(test)]
mod test_power;
#[cfg(test)]
mod test_subtract;
#[cfg(test)]
mod test_tail_to_nan;
#[cfg(test)]
mod test_tradewhen;
#[cfg(test)]
mod test_ts_argmax;
#[cfg(test)]
mod test_ts_argmin;
#[cfg(test)]
mod test_ts_argminmax;
#[cfg(test)]
mod test_ts_decay_exp;
#[cfg(test)]
mod test_ts_decay_linear;
#[cfg(test)]
mod test_ts_delay;
#[cfg(test)]
mod test_ts_diff;
#[cfg(test)]
mod test_ts_ffill;
#[cfg(test)]
mod test_ts_mae;
#[cfg(test)]
mod test_ts_max;
#[cfg(test)]
mod test_ts_mean;
#[cfg(test)]
mod test_ts_min;
#[cfg(test)]
mod test_ts_rank;
#[cfg(test)]
mod test_ts_std;
#[cfg(test)]
mod test_ts_sum;
#[cfg(test)]
mod test_ts_zscore;
#[cfg(test)]
mod test_where;

use std::collections::VecDeque;

pub trait PartialDeque {
    fn push(&mut self, new_data: &[f64]);
}

/// A generic-purpose ring buffer structured around VecDeque for safe Rust-native operations.
/// Rows = Time (capacity), Cols = Cross-section (len).
#[derive(Debug, Clone)]
pub struct DequeState {
    pub history: VecDeque<Vec<f64>>,
    pub col: usize,
    pub cap: usize,
    pub count: usize,
}

impl DequeState {
    pub fn new(capacity: usize, length: usize) -> Self {
        DequeState {
            history: VecDeque::with_capacity(if capacity == 0 { 1 } else { capacity + 1 }),
            col: length,
            cap: capacity,
            count: 0,
        }
    }

    /// Gets the oldest row (the one that will be overwritten next).
    pub fn get_oldest(&self) -> Option<&[f64]> {
        if self.count < self.cap || self.cap == 0 {
            return None;
        }
        self.history.front().map(|v| v.as_slice())
    }

    pub fn get_latest(&self) -> Option<&[f64]> {
        if self.count == 0 {
            return None;
        }
        self.history.back().map(|v| v.as_slice())
    }

    pub fn get_latest_mut(&mut self) -> Option<&mut [f64]> {
        if self.count == 0 {
            return None;
        }
        self.history.back_mut().map(|v| v.as_mut_slice())
    }
}

impl PartialDeque for DequeState {
    fn push(&mut self, new_data: &[f64]) {
        if self.cap == 0 || self.col == 0 {
            return;
        }
        assert_eq!(
            new_data.len(),
            self.col,
            "new_data len must match state col"
        );

        if self.count < self.cap {
            self.count += 1;
            self.history.push_back(new_data.to_vec());
        } else {
            // Re-use the oldest allocation for performance zero-copy where possible
            let mut oldest = self.history.pop_front().unwrap();
            oldest.copy_from_slice(new_data);
            self.history.push_back(oldest);
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Constant = 0,
    TimeSeries = 1,
    DataFrame = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CometData {
    pub dtype: DataType,
    pub ptr: *const f64,
}

impl CometData {
    #[inline(always)]
    pub unsafe fn as_slice(&self, len: usize) -> &[f64] {
        if self.dtype == DataType::DataFrame {
            unsafe { std::slice::from_raw_parts(self.ptr, len) }
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, 1) }
        }
    }

    #[inline(always)]
    pub unsafe fn get_scalar(&self) -> f64 {
        unsafe { *self.ptr }
    }
}

// In src/stdlib/lib.rs
pub trait ConstOp {
    fn new(c: f64, len: usize) -> Self;
    fn step(&mut self, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}

pub trait UnaryOp {
    fn new(period: usize, len: usize) -> Self;
    fn step(&mut self, a: CometData, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}

pub trait BinaryOp {
    fn new(period: usize, len: usize) -> Self;
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}

pub trait TernaryOp {
    fn new(period: usize, len: usize) -> Self;
    fn step(&mut self, a: CometData, b: CometData, c: CometData, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}
