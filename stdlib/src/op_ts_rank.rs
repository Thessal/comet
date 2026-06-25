use crate::{OperatorSpec, types::Signal};

pub static OP_TS_RANK: OperatorSpec = OperatorSpec {
    name: "ts_rank",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            let t_len = a.size()[0];
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(tch::Tensor::full(a.size().as_slice(), 1.0, (a.kind(), a.device()))));
            }

            let res = tch::Tensor::empty(a.size().as_slice(), (a.kind(), a.device()));
            for t in 0..t_len {
                let start = std::cmp::max(0, t - d + 1);
                let slice = a.narrow(0, start, t - start + 1);
                let last_elem = a.narrow(0, t, 1);
                
                let less_equal = slice.less_equal_tensor(&last_elem);
                let valid = slice.isnan().logical_not();
                let valid_less_equal = less_equal.logical_and(&valid).to_kind(a.kind());
                
                let rank = valid_less_equal.sum_dim_intlist(Some(&[0][..]), false, a.kind());
                let count_valid = valid.to_kind(a.kind()).sum_dim_intlist(Some(&[0][..]), false, a.kind());
                let step_res = &rank / &count_valid;
                
                let is_last_nan = last_elem.isnan().squeeze_dim(0);
                let nan = tch::Tensor::full(step_res.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                let step_res = step_res.where_self(&is_last_nan.logical_not(), &nan);
                
                let mut row = res.narrow(0, t, 1);
                let _ = row.copy_(&step_res.unsqueeze(0));
            }
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
