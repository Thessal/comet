use crate::{OperatorSpec, types::Signal};

pub static OP_TS_MEAN: OperatorSpec = OperatorSpec {
    name: "ts_mean",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(t))) => {
            let t = std::cmp::max(1, *t as i64);
            let t_len = a.size()[0];

            let a_clean = a.nan_to_num(0.0, 0.0, 0.0);
            let cumsum = a_clean.cumsum(0, a.kind());

            let mut zeros_shape = a.size();
            zeros_shape[0] = t;
            let zeros = tch::Tensor::zeros(&zeros_shape, (a.kind(), a.device()));

            let sub = if t_len > t {
                tch::Tensor::cat(&[&zeros, &cumsum.narrow(0, 0, t_len - t)], 0)
            } else {
                let mut full_zeros_shape = a.size();
                full_zeros_shape[0] = t_len;
                tch::Tensor::zeros(&full_zeros_shape, (a.kind(), a.device()))
            };

            let sum = &cumsum - &sub;

            let is_valid = a.isnan().logical_not().to_kind(a.kind());
            let count_cumsum = is_valid.cumsum(0, a.kind());

            let count_sub = if t_len > t {
                tch::Tensor::cat(&[&zeros, &count_cumsum.narrow(0, 0, t_len - t)], 0)
            } else {
                let mut full_zeros_shape = a.size();
                full_zeros_shape[0] = t_len;
                tch::Tensor::zeros(&full_zeros_shape, (a.kind(), a.device()))
            };

            let count = &count_cumsum - &count_sub;

            let mean = sum / &count;
            
            let is_zero = count.eq(0.0);
            let nan = tch::Tensor::full(mean.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let res = mean.where_self(&is_zero.logical_not(), &nan);

            Signal::DataFrame(Some(res))
        }
        _ => panic!("ts_mean expected DataFrame and Int"),
    },
};

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use tch::Tensor;

//     #[test]
//     fn test_ts_mean() {
//         let a = Tensor::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]).view([5, 1]);
//         let out =
//             (OP_TS_MEAN.execute)(&[Signal::DataFrame(Some(a)), Signal::Int(Some(3))]).unwrap();
//         if let Signal::DataFrame(Some(res)) = out {
//             res.print();
//             let expected = Tensor::from_slice(&[2.0, 3.0, 4.0, 6.0, 8.0]).view([5, 1]);
//             assert!(bool::from(res.isclose(&expected, 1e-5, 1e-8, false).all()));
//         } else {
//             panic!("Wrong output");
//         }
//     }
// }
