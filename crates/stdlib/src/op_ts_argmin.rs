use crate::{OperatorSpec, types::Signal};

pub static OP_TS_ARGMIN: OperatorSpec = OperatorSpec {
    name: "ts_argmin",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 0 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }
            let window_size = d + 1;

            let w = crate::op_time_series::roll_window(a, window_size);
            let reversed = w.flip([-1]);
            let filled = reversed.nan_to_num(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY);
            let step_argmin = filled.argmin(-1, false).to_kind(a.kind());
            
            let all_nan = w.isnan().all_dim(-1, false);
            let nan = tch::Tensor::full(step_argmin.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let res = step_argmin.where_self(&all_nan.logical_not(), &nan);
            
            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_argmin expected DataFrame and Int"),
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_ts_argmin() {
        let a = Tensor::from_slice(&[3.0, 1.0, f64::NAN, 2.0, 5.0]).view([5, 1]);
        let out = (OP_TS_ARGMIN.execute)(&[Signal::DataFrame(Some(a)), Signal::Int(Some(2))]);
        if let Signal::DataFrame(Some(res)) = out {
            let expected = Tensor::from_slice(&[0.0, 0.0, 1.0, 2.0, 1.0]).view([5, 1]);
            let is_all_true = i64::try_from(res.isclose(&expected, 1e-5, 1e-8, true).all()).unwrap() != 0;
            assert!(is_all_true);
        } else {
            panic!("Wrong output");
        }
    }
}
