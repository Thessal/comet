use crate::CometData;

#[repr(C)]
pub struct ClipState {
    pub lower: f64,
    pub upper: f64,
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_init(lower: f64, upper: f64, _len: usize) -> *mut ClipState {
    let state = Box::new(ClipState { lower, upper });
    Box::into_raw(state)
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_free(state: *mut ClipState) {
    if !state.is_null() {
        unsafe {
            let _ = Box::from_raw(state);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_step(
    state: *mut ClipState,
    signal: *const CometData,
    out_ptr: *mut f64,
    len: usize
) {
    let s = unsafe { &mut *state };
    let sig_val = unsafe { *signal };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

    if sig_val.dtype == crate::DataType::DataFrame {
        let sig_slice = unsafe { sig_val.as_slice(len) };
        for i in 0..len {
            out[i] = sig_slice[i].clamp(s.lower, s.upper);
        }
    } else {
        out[0] = unsafe { sig_val.get_scalar() }.clamp(s.lower, s.upper);
    }
}
