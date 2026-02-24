use crate::RingBufferF64;

#[repr(C)]
pub struct CovarianceState {
    pub history: RingBufferF64,
    pub period: usize,
    pub time: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_covariance_init(period: usize, len: usize) -> *mut CovarianceState {
    let state = Box::new(CovarianceState {
        history: RingBufferF64::new(period, len),
        period,
        time: 0,
    });
    Box::into_raw(state)
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_covariance_free(state: *mut CovarianceState) {
    if !state.is_null() {
        unsafe { 
            let mut s = Box::from_raw(state); 
            s.history.drop_inner(); 
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_covariance_step(
    state: *mut CovarianceState,
    returns_ptr: *const crate::CometData,
    _lookback: usize,
    out_ptr: *mut f64,
    len: usize
) {
    let s = unsafe { &mut *state };
    let returns = unsafe { (*returns_ptr).as_slice(len) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_ptr, len * len) };

    s.history.push(returns);
    s.time += 1;

    if s.time < s.period {
        for i in 0..(len * len) { out[i] = f64::NAN; }
        return;
    }

    let buf_ptr = s.history.ptr;
    let cap = s.history.cap;

    let mut means = vec![0.0; len];
    let mut counts = vec![0.0; len];

    for t in 0..cap {
        let row_start = t * len;
        let row = unsafe { std::slice::from_raw_parts(buf_ptr.add(row_start), len) };
        for i in 0..len {
            if !row[i].is_nan() {
                means[i] += row[i];
                counts[i] += 1.0;
            }
        }
    }
    
    for i in 0..len {
        if counts[i] > 0.0 {
            means[i] /= counts[i];
        } else {
            means[i] = f64::NAN;
        }
    }

    for i in 0..len {
        for j in 0..len {
            let out_idx = i * len + j;
            if means[i].is_nan() || means[j].is_nan() {
                out[out_idx] = f64::NAN;
                continue;
            }

            let mut cov = 0.0;
            let mut pair_count = 0.0;

            for t in 0..cap {
                let row_start = t * len;
                let row = unsafe { std::slice::from_raw_parts(buf_ptr.add(row_start), len) };
                if !row[i].is_nan() && !row[j].is_nan() {
                    cov += (row[i] - means[i]) * (row[j] - means[j]);
                    pair_count += 1.0;
                }
            }

            if pair_count > 1.0 {
                out[out_idx] = cov / (pair_count - 1.0);
            } else {
                out[out_idx] = f64::NAN;
            }
        }
    }
}
