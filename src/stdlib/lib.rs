// src/stdlib/lib.rs

pub mod add;
pub mod ts_mean;

#[cfg(test)]
mod test_ts_mean;

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

// In src/stdlib/lib.rs
pub trait StatefulUnary {
    fn new(period: usize, len: usize) -> Self;
    fn step(&mut self, a_ptr: *const f64, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}

pub trait StatefulBinary {
    fn new(period: usize, len: usize) -> Self;
    fn step(&mut self, a_ptr: *const f64, b_ptr: *const f64, out_ptr: *mut f64, len: usize);
    fn drop_buffers(&mut self) {}
}

#[macro_export]
macro_rules! export_unary {
    ($struct_name:ident, $prefix:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _init>](period: usize, len: usize) -> *mut $struct_name {
                let state = Box::new($struct_name::new(period, len));
                Box::into_raw(state)
            }
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _free>](state: *mut $struct_name) {
                if !state.is_null() {
                    unsafe {
                        let mut s = Box::from_raw(state);
                        s.drop_buffers();
                    }
                }
            }
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _step>](
                state: *mut $struct_name, 
                a_ptr: *const f64, 
                out_ptr: *mut f64, 
                len: usize
            ) {
                let s = unsafe { &mut *state };
                s.step(a_ptr, out_ptr, len)
            }
        }
    };
}

#[macro_export]
macro_rules! export_binary {
    ($struct_name:ident, $prefix:ident) => {
        paste::paste! {
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _init>](period: usize, len: usize) -> *mut $struct_name {
                let state = Box::new($struct_name::new(period, len));
                Box::into_raw(state)
            }
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _free>](state: *mut $struct_name) {
                if !state.is_null() {
                    unsafe {
                        let mut s = Box::from_raw(state);
                        s.drop_buffers();
                    }
                }
            }
            #[unsafe(no_mangle)]
            pub extern "C" fn [<comet_ $prefix _step>](
                state: *mut $struct_name, 
                a_ptr: *const f64, 
                b_ptr: *const f64, 
                out_ptr: *mut f64, 
                len: usize
            ) {
                let s = unsafe { &mut *state };
                s.step(a_ptr, b_ptr, out_ptr, len)
            }
        }
    };
}