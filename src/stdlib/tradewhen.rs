use crate::{CometData, DataType};

#[repr(C)]
pub struct TradewhenState {
    pub last_val: Vec<f64>,
    pub time_since: Vec<usize>,
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_tradewhen_init(_period: usize, len: usize) -> *mut TradewhenState {
    let state = Box::new(TradewhenState {
        last_val: vec![f64::NAN; len],
        time_since: vec![usize::MAX; len],
    });
    Box::into_raw(state)
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_tradewhen_free(state: *mut TradewhenState) {
    if !state.is_null() {
        unsafe { let _ = Box::from_raw(state); }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_tradewhen_step(
    state: *mut TradewhenState,
    signal: *const CometData,
    enter: *const CometData,
    exit: *const CometData,
    period: usize,
    out_ptr: *mut f64,
    len: usize
) {
    let s = unsafe { &mut *state };
    let sig_val = unsafe { *signal };
    let ent_val = unsafe { *enter };
    let ext_val = unsafe { *exit };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };

    let is_sig_df = sig_val.dtype == DataType::DataFrame;
    let is_ent_df = ent_val.dtype == DataType::DataFrame;
    let is_ext_df = ext_val.dtype == DataType::DataFrame;

    let sig_scalar = if !is_sig_df { unsafe { sig_val.get_scalar() } } else { f64::NAN };
    let ent_scalar = if !is_ent_df { unsafe { ent_val.get_scalar() } } else { 0.0 };
    let ext_scalar = if !is_ext_df { unsafe { ext_val.get_scalar() } } else { 0.0 };

    for i in 0..len {
        let sig = if is_sig_df { unsafe { sig_val.as_slice(len)[i] } } else { sig_scalar };
        let ent = if is_ent_df { unsafe { ent_val.as_slice(len)[i] } } else { ent_scalar };
        let ext = if is_ext_df { unsafe { ext_val.as_slice(len)[i] } } else { ext_scalar };

        let enter_cond = ent > 0.0 && !ent.is_nan();
        let exit_cond = ext > 0.0 && !ext.is_nan() && !enter_cond;

        let mut input_val = std::f64::NAN;
        if enter_cond {
            input_val = sig;
        } else if exit_cond {
            input_val = std::f64::INFINITY;
        }

        // Apply ffill logic on input_val
        if !input_val.is_nan() {
            s.last_val[i] = input_val;
            s.time_since[i] = 0;
        } else {
            if s.time_since[i] != usize::MAX {
                s.time_since[i] = s.time_since[i].saturating_add(1);
            }
        }

        if s.time_since[i] <= period {
            if s.last_val[i].is_infinite() {
                out[i] = f64::NAN;
            } else {
                out[i] = s.last_val[i];
            }
        } else {
            out[i] = f64::NAN;
        }
    }
}
