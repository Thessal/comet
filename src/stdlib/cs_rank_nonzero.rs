use crate::CometData;
use crate::cs_rank::CsRankState; //step as cs_rank_step;
use crate::{UnaryOp, export_unary};

// TODO: use `export_unary` macro

#[repr(C)]
pub struct CsRankNonzeroState {
    pub eps: f64,
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_cs_rank_nonzero_init(eps: f64, _len: usize) -> *mut CsRankNonzeroState {
    let state = Box::new(CsRankNonzeroState { eps });
    Box::into_raw(state)
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_cs_rank_nonzero_free(state: *mut CsRankNonzeroState) {
    if !state.is_null() {
        unsafe {
            let _ = Box::from_raw(state);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_cs_rank_nonzero_step(
    state: *mut CsRankNonzeroState,
    signal: *const CometData,
    out_ptr: *mut f64,
    len: usize
) {
    let mut csrank_state = crate::cs_rank::CsRankState{};
    csrank_state.step(unsafe { *signal }, out_ptr, len);
    let eps = unsafe { (*state).eps };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
    match unsafe { (*signal).dtype } {
        crate::DataType::DataFrame => {
            for i in 0..len {
                out[i] += eps;
            }
        },
        crate::DataType::TimeSeries => {
            out[0] += eps;
        },
        crate::DataType::Constant => {
            out[0] += eps;
        },
    }
}

//export_unary!(CsRankNonzeroState, cs_rank_nonzero);
