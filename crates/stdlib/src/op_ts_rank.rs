use crate::{OperatorSpec, types::Signal};

pub static OP_TS_RANK: OperatorSpec = OperatorSpec {
    name: "ts_rank",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(tch::Tensor::full(a.size().as_slice(), 1.0, (a.kind(), a.device()))));
            }

            let w = crate::op_time_series::roll_window(a, d);
            let a_unsqueezed = a.unsqueeze(2);
            let less_equal = w.less_equal_tensor(&a_unsqueezed);
            
            let valid = w.isnan().logical_not();
            let valid_less_equal = less_equal.logical_and(&valid).to_kind(a.kind());
            
            let rank = valid_less_equal.sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let count_valid = valid.to_kind(a.kind()).sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let step_res = &rank / &count_valid;
            
            let is_last_nan = a.isnan();
            let nan = tch::Tensor::full(step_res.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let res = step_res.where_self(&is_last_nan.logical_not(), &nan);
            
            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_rank expected DataFrame and Int"),
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_ts_rank() {
        let a = Tensor::from_slice(&[1.0, 3.0, 2.0, f64::NAN, 5.0]).view([5, 1]);
        let out = (OP_TS_RANK.execute)(&[Signal::DataFrame(Some(a)), Signal::Int(Some(3))]);
        if let Signal::DataFrame(Some(res)) = out {
            let expected = Tensor::from_slice(&[1.0, 1.0, 2.0/3.0, f64::NAN, 1.0]).view([5, 1]);
            let is_all_true = i64::try_from(res.isclose(&expected, 1e-5, 1e-8, true).all()).unwrap() != 0;
            assert!(is_all_true);
        } else {
            panic!("Wrong output");
        }
    }
}
