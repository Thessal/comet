use crate::ZeroAryOp;
use std::slice;

#[repr(C)]
pub struct ConstState {
    pub c: f64,
    pub len: usize,
}

impl ConstState {
    pub fn new(c: f64, len: usize) -> Self {
        ConstState { c, len }
    }
}
impl ZeroAryOp for ConstState {
    fn step(&mut self, out_ptr: *mut f64) {
        let out = unsafe { slice::from_raw_parts_mut(out_ptr, 1) };
        out[0] = self.c;
    }
}

// #[unsafe(no_mangle)]
// pub extern "C" fn comet_const_init(c: f64, len: usize) -> *mut ConstState {
//     let state = Box::new(ConstState::new(c, len));
//     Box::into_raw(state)
// }
// #[unsafe(no_mangle)]
// pub extern "C" fn comet_const_free(state: *mut ConstState) {
//     if !state.is_null() {
//         unsafe {
//             let mut s = Box::from_raw(state);
//             s.drop_buffers();
//         }
//     }
// }
// #[unsafe(no_mangle)]
// pub extern "C" fn comet_const_step(state: *mut ConstState, out_ptr: *mut f64, len: usize) {
//     let s = unsafe { &mut *state };
//     s.step(out_ptr)
// }

inventory::submit! {
    crate::OperatorMeta {
        name: "const",
        output_shape: crate::OutputShape::TimeSeries,
    }
}
