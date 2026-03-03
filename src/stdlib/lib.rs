// src/stdlib/lib.rs

pub mod add;
pub mod subtract;
pub mod multiply;
pub mod power;
pub mod equals;
pub mod greater;
pub mod less;
pub mod min;
pub mod max;
pub mod cs_zscore;
pub mod cs_rank;
pub mod ts_mean;
pub mod ts_sum;
pub mod ts_min;
pub mod ts_max;
pub mod ts_rank;
pub mod clip;
pub mod tail_to_nan;
pub mod r#where;
pub mod tradewhen;
pub mod covariance;
pub mod ts_diff;
pub mod abs;
pub mod ts_decay_linear;
pub mod ts_decay_exp;
pub mod ts_argmin;
pub mod ts_argmax;
pub mod ts_argminmax;
pub mod ts_ffill;
pub mod ts_delay;
pub mod ts_std;
pub mod ts_mae;
pub mod ts_zscore;
pub mod mid;
pub mod r#const;
pub mod divide;
pub mod data;
pub mod consume;
pub mod cs_rank_nonzero;

#[cfg(test)]
mod test_ts_mean;
#[cfg(test)]
mod test_cs_zscore;
#[cfg(test)]
mod test_cs_rank;
#[cfg(test)]
mod test_abs;
#[cfg(test)]
mod test_ts_diff;
#[cfg(test)]
mod test_ts_decay_linear;
#[cfg(test)]
mod test_ts_decay_exp;
#[cfg(test)]
mod test_ts_argmin;
#[cfg(test)]
mod test_ts_argmax;
#[cfg(test)]
mod test_ts_argminmax;
#[cfg(test)]
mod test_ts_ffill;
#[cfg(test)]
mod test_ts_delay;
#[cfg(test)]
mod test_ts_std;
#[cfg(test)]
mod test_ts_mae;
#[cfg(test)]
mod test_ts_zscore;
#[cfg(test)]
mod test_mid;
#[cfg(test)]
mod test_const;
#[cfg(test)]
mod test_divide;
#[cfg(test)]
mod test_subtract;
#[cfg(test)]
mod test_multiply;
#[cfg(test)]
mod test_power;
#[cfg(test)]
mod test_equals;
#[cfg(test)]
mod test_greater;
#[cfg(test)]
mod test_less;
#[cfg(test)]
mod test_min;
#[cfg(test)]
mod test_max;
#[cfg(test)]
mod test_ts_sum;
#[cfg(test)]
mod test_ts_min;
#[cfg(test)]
mod test_ts_max;
#[cfg(test)]
mod test_ts_rank;
#[cfg(test)]
mod test_clip;
#[cfg(test)]
mod test_tail_to_nan;
#[cfg(test)]
mod test_where;
#[cfg(test)]
mod test_tradewhen;
#[cfg(test)]
mod test_covariance;

/// A generic-purpose 2D ring buffer specifically designed for C-ABI / LLVM compatibility.
/// Rows = Time (capacity), Cols = Cross-section (len).
#[repr(C)]
pub struct RingBufferF64 {
    pub ptr: *mut f64,    // Pointer to heap-allocated flat 2D buffer
    pub cap: usize,       // Maximum time periods
    pub len: usize,       // Number of features / dimensions per time period
    pub head: usize,      // Current insert row position
    pub count: usize,     // Total rows in buffer (up to cap)
}

impl RingBufferF64 {
    pub fn new(capacity: usize, length: usize) -> Self {
        let total_size = capacity * length;
        let mut buf = vec![0.0; total_size];
        let ptr = buf.as_mut_ptr();
        std::mem::forget(buf); 
        
        RingBufferF64 {
            ptr,
            cap: capacity,
            len: length,
            head: 0,
            count: 0,
        }
    }

    /// Pushes a new row slice into the buffer. 
    pub fn push(&mut self, val_row: &[f64]) {
        if self.cap == 0 || self.len == 0 {
            return;
        }

        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.cap * self.len) };
        let row_start = self.head * self.len;
        let row_end = row_start + self.len;
        
        if self.count < self.cap {
            self.count += 1;
        }
        
        // Copy the new row in
        slice[row_start..row_end].copy_from_slice(val_row);
        self.head = (self.head + 1) % self.cap;
    }

    /// Gets the oldest row (the one that will be overwritten next).
    pub fn get_oldest<'a>(&self) -> Option<&'a [f64]> {
        if self.count < self.cap || self.cap == 0 {
            return None;
        }
        let slice = unsafe { std::slice::from_raw_parts(self.ptr, self.cap * self.len) };
        let row_start = self.head * self.len;
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr().add(row_start), self.len) })
    }

    pub fn get_latest<'a>(&self) -> Option<&'a [f64]> {
        if self.count == 0 {
            return None;
        }
        let latest_idx = if self.head == 0 { self.cap - 1 } else { self.head - 1 };
        let slice = unsafe { std::slice::from_raw_parts(self.ptr, self.cap * self.len) };
        let row_start = latest_idx * self.len;
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr().add(row_start), self.len) })
    }

    pub fn get_latest_mut<'a>(&mut self) -> Option<&'a mut [f64]> {
        if self.count == 0 {
            return None;
        }
        let latest_idx = if self.head == 0 { self.cap - 1 } else { self.head - 1 };
        let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr, self.cap * self.len) };
        let row_start = latest_idx * self.len;
        Some(unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr().add(row_start), self.len) })
    }

    /// Safely frees the allocated buffer
    pub fn drop_inner(&mut self) {
        let total = self.cap * self.len;
        if !self.ptr.is_null() && total > 0 {
            unsafe {
                let _ = Vec::from_raw_parts(self.ptr, total, total);
            }
            self.ptr = std::ptr::null_mut();
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