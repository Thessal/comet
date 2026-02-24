use crate::{CometData, DataType};

#[unsafe(no_mangle)]
pub extern "C" fn comet_where_init() -> *mut std::ffi::c_void {
    std::ptr::null_mut()
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_where_free(_state: *mut std::ffi::c_void) {}

#[unsafe(no_mangle)]
pub extern "C" fn comet_where_step(
    _state: *mut std::ffi::c_void,
    condition: *const CometData,
    val_true: *const CometData,
    val_false: *const CometData,
    out_ptr: *mut f64,
    len: usize
) {
    let cond_val = unsafe { *condition };
    let t_val = unsafe { *val_true };
    let f_val = unsafe { *val_false };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

    let is_cond_df = cond_val.dtype == DataType::DataFrame;
    let is_t_df = t_val.dtype == DataType::DataFrame;
    let is_f_df = f_val.dtype == DataType::DataFrame;

    let cond_scalar = if !is_cond_df { unsafe { cond_val.get_scalar() } } else { 0.0 };
    let t_scalar = if !is_t_df { unsafe { t_val.get_scalar() } } else { 0.0 };
    let f_scalar = if !is_f_df { unsafe { f_val.get_scalar() } } else { 0.0 };

    for i in 0..len {
        let c = if is_cond_df { unsafe { cond_val.as_slice(len)[i] } } else { cond_scalar };
        let t = if is_t_df { unsafe { t_val.as_slice(len)[i] } } else { t_scalar };
        let f = if is_f_df { unsafe { f_val.as_slice(len)[i] } } else { f_scalar };

        if c > 0.0 {
            out[i] = t;
        } else if c < 0.0 {
            out[i] = f;
        } else {
            out[i] = f64::NAN;
        }
    }
}
