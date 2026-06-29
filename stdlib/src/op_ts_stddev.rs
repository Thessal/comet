use crate::{OperatorSpec, types::Signal};

pub static OP_TS_STDDEV: OperatorSpec = OperatorSpec {
    name: "ts_stddev",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 2 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }

            let w = crate::op_time_series::roll_window(a, d);
            
            let valid = w.isnan().logical_not();
            let count_valid = valid.to_kind(a.kind()).sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            
            let nansum = w.nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let mean = &nansum / &count_valid;
            
            let diff = &w - mean.unsqueeze(-1);
            let var = diff.square().nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(Some(&[-1][..]), false, a.kind()) / (&count_valid - 1.0);
            let stddev = var.sqrt();
            
            let is_invalid = count_valid.less_equal(1.0);
            let nan = tch::Tensor::full(stddev.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let res = stddev.where_self(&is_invalid.logical_not(), &nan);
            
            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_stddev expected DataFrame and Int"),
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_ts_stddev() {
        let a = Tensor::from_slice(&[2.0, 4.0, f64::NAN, 8.0, 10.0]).view([5, 1]);
        let out = (OP_TS_STDDEV.execute)(&[Signal::DataFrame(Some(a)), Signal::Int(Some(3))]);
        if let Signal::DataFrame(Some(res)) = out {
            let std_24 = 1.41421356237;
            let std_48 = 2.82842712475;
            let std_810 = 1.41421356237;
            let expected = Tensor::from_slice(&[f64::NAN, std_24, std_24, std_48, std_810]).view([5, 1]);
            let is_all_true = i64::try_from(res.isclose(&expected, 1e-4, 1e-6, true).all()).unwrap() != 0;
            assert!(is_all_true);
        } else {
            panic!("Wrong output");
        }
    }
}
