use crate::{OperatorSpec, types::Signal};

pub static OP_TS_SUM: OperatorSpec = OperatorSpec {
    name: "ts_sum",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }

            let w = crate::op_time_series::roll_window(a, d);
            let nansum = w.nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let all_nan = w.isnan().all_dim(-1, false);
            let nan = tch::Tensor::full(nansum.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let res = nansum.where_self(&all_nan.logical_not(), &nan);
            
            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_sum expected DataFrame and Int"),
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_ts_sum() {
        let a = Tensor::from_slice(&[1.0, f64::NAN, 3.0, 2.0, 5.0]).view([5, 1]);
        let out = (OP_TS_SUM.execute)(&[Signal::DataFrame(Some(a)), Signal::Int(Some(3))]);
        if let Signal::DataFrame(Some(res)) = out {
            let expected = Tensor::from_slice(&[1.0, 1.0, 4.0, 5.0, 10.0]).view([5, 1]);
            let is_all_true = i64::try_from(res.isclose(&expected, 1e-5, 1e-8, true).all()).unwrap() != 0;
            assert!(is_all_true);
        } else {
            panic!("Wrong output");
        }
    }
}
