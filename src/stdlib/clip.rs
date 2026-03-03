use crate::CometData;

#[repr(C)]
pub struct ClipState {
    pub lower: f64,
    pub upper: f64,
}

impl ClipState {
    pub fn new(lower: f64, upper: f64, _len: usize) -> Self {
        ClipState { lower, upper }
    }

    pub fn step(&mut self, signal: CometData, out_ptr: *mut f64, len: usize) {
        let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        if signal.dtype == crate::DataType::DataFrame {
            let sig_slice = unsafe { signal.as_slice(len) };
            for i in 0..len {
                out[i] = sig_slice[i].clamp(self.lower, self.upper);
            }
        } else {
            out[0] = unsafe { signal.get_scalar() }.clamp(self.lower, self.upper);
        }
    }
}
