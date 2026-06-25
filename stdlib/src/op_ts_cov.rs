use crate::{OperatorSpec, types::Signal};

pub static OP_TS_COV: OperatorSpec = OperatorSpec {
    name: "ts_cov",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1], &args[2]) {
        (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            let t_len = x.size()[0];
            if d < 2 {
                let nan = tch::Tensor::full(x.size().as_slice(), f64::NAN, (x.kind(), x.device()));
                return Signal::DataFrame(Some(nan));
            }

            let res = tch::Tensor::empty(x.size().as_slice(), (x.kind(), x.device()));
            for t in 0..t_len {
                let start = std::cmp::max(0, t - d + 1);
                let slice_x = x.narrow(0, start, t - start + 1);
                let slice_y = y.narrow(0, start, t - start + 1);
                
                let valid_x = slice_x.isnan().logical_not();
                let valid_y = slice_y.isnan().logical_not();
                let valid_both = valid_x.logical_and(&valid_y);
                let count_both = valid_both.to_kind(x.kind()).sum_dim_intlist(Some(&[0][..]), false, x.kind());
                
                let zeros = tch::Tensor::zeros(&[1], (x.kind(), x.device()));
                let x_masked = slice_x.where_self(&valid_both, &zeros);
                let y_masked = slice_y.where_self(&valid_both, &zeros);
                
                let mean_x = x_masked.sum_dim_intlist(Some(&[0][..]), false, x.kind()) / &count_both;
                let mean_y = y_masked.sum_dim_intlist(Some(&[0][..]), false, x.kind()) / &count_both;
                
                let diff_x = (&x_masked - mean_x.unsqueeze(0)).where_self(&valid_both, &zeros);
                let diff_y = (&y_masked - mean_y.unsqueeze(0)).where_self(&valid_both, &zeros);
                
                let cov = (&diff_x * &diff_y).sum_dim_intlist(Some(&[0][..]), false, x.kind()) / (&count_both - 1.0);
                
                let is_invalid = count_both.less_equal(1.0);
                let nan = tch::Tensor::full(cov.size().as_slice(), f64::NAN, (x.kind(), x.device()));
                let step_res = cov.where_self(&is_invalid.logical_not(), &nan);
                
                let mut row = res.narrow(0, t, 1);
                let _ = row.copy_(&step_res.unsqueeze(0));
            }
            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_cov expected DataFrame, DataFrame and Int"),
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_ts_cov() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]).view([4, 1]);
        let y = Tensor::from_slice(&[1.0, 2.0, 1.0, 2.0]).view([4, 1]);
        let out = (OP_TS_COV.execute)(&[Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y)), Signal::Int(Some(3))]);
        if let Signal::DataFrame(Some(res)) = out {
            assert_eq!(res.size(), vec![4, 1]);
        } else {
            panic!("Wrong output");
        }
    }
}
