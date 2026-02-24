use crate::CometData;

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_init() -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_free(_state: *mut std::ffi::c_void) {}

#[unsafe(no_mangle)]
pub extern "C" fn comet_clip_step(
    _state: *mut std::ffi::c_void,
    signal: *const CometData,
    lower: f64,
    upper: f64,
    out_ptr: *mut f64,
    len: usize
) {
    let sig_val = unsafe { *signal };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

    if sig_val.dtype == crate::DataType::DataFrame {
        let sig_slice = unsafe { sig_val.as_slice(len) };
        for i in 0..len {
            out[i] = sig_slice[i].clamp(lower, upper);
        }
    } else {
        out[0] = unsafe { sig_val.get_scalar() }.clamp(lower, upper);
    }
}
