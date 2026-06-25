use crate::{OperatorSpec, types::Signal};

pub static OP_TS_ARGMIN: OperatorSpec = OperatorSpec {
    name: "ts_argmin",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            let t_len = a.size()[0];
            if d < 0 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }
            let window_size = d + 1;
            let res = tch::Tensor::empty(a.size().as_slice(), (a.kind(), a.device()));
            for t in 0..t_len {
                let start = std::cmp::max(0, t - window_size + 1);
                let slice = a.narrow(0, start, t - start + 1);
                let reversed = slice.flip([0]);
                let pad_len = window_size - reversed.size()[0];
                let filled = reversed.nan_to_num(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY);
                let padded = if pad_len > 0 {
                    let mut pad_shape = filled.size();
                    pad_shape[0] = pad_len;
                    let inf_pad = tch::Tensor::full(&pad_shape, std::f64::INFINITY, (a.kind(), a.device()));
                    tch::Tensor::cat(&[&filled, &inf_pad], 0)
                } else {
                    filled
                };
                let step_argmin = padded.argmin(0, false).to_kind(a.kind());
                let all_nan = slice.isnan().all_dim(0, false);
                let nan = tch::Tensor::full(step_argmin.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                let step_res = step_argmin.where_self(&all_nan.logical_not(), &nan);
                let mut row = res.narrow(0, t, 1);
                let _ = row.copy_(&step_res.unsqueeze(0));
            }
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
