use crate::{DequeState, PartialDeque, UnaryOp};
#[repr(C)]
pub struct TsDelayState {
    pub history: DequeState,
    pub len: usize,
}

// 1. Implement the generic trait
impl TsDelayState {
    pub fn new(period: usize, len: usize) -> Self {
        TsDelayState {
            history: DequeState::new(period, len), len }
    }
}
impl UnaryOp for TsDelayState {
    fn step(&mut self, a: crate::CometData, out_ptr: *mut f64) {
        let len = self.len;
        let a_slice = unsafe { a.as_slice(len) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

        // 1. Get the oldest slice before overwriting it
        let old_history_slice_opt = self.history.get_oldest();

        for i in 0..len {
            // 2. Erase the oldest value if we reached capacity and it wasn't NaN
            if let Some(old_history_slice) = old_history_slice_opt {
                let old_val = old_history_slice[i];
                out_slice[i] = old_val;
            } else {
                out_slice[i] = f64::NAN;
            }
        }

        // 5. Finally, push the new slice into history memory, overwriting the oldest value.
        self.history.push(a_slice);
    }

    fn drop_buffers(&mut self) {}
}


inventory::submit! {
    crate::OperatorMeta {
        name: "ts_delay",
        output_shape: crate::OutputShape::DataFrame,
    }
}
