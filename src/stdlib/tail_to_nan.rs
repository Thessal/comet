use crate::CometData;

#[unsafe(no_mangle)]
pub extern "C" fn comet_tail_to_nan_init() -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_tail_to_nan_free(_state: *mut std::ffi::c_void) {}

#[unsafe(no_mangle)]
pub extern "C" fn comet_tail_to_nan_step(
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
            let v = sig_slice[i];
            out[i] = if v < lower || v > upper { f64::NAN } else { v };
        }
    } else {
        let v = unsafe { sig_val.get_scalar() };
        out[0] = if v < lower || v > upper { f64::NAN } else { v };
    }
}
