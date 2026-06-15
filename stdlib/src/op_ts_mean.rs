use crate::{OperatorSpec, types::Signal};

pub static OP_TS_MEAN: OperatorSpec = OperatorSpec {
    name: "ts_mean",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(t))) => {
            let t = std::cmp::max(1, *t as i64);
            let t_len = a.size()[0];

            let cumsum = a.cumsum(0, a.kind());

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

            let div_vec: Vec<f64> = (0..t_len).map(|i| std::cmp::min(i + 1, t) as f64).collect();
            let mut div_shape = vec![1; a.size().len()];
            div_shape[0] = t_len;
            let divisor = tch::Tensor::from_slice(&div_vec)
                .to_device(a.device())
                .to_kind(a.kind())
                .view(div_shape.as_slice());

            Signal::DataFrame(Some(sum / divisor))
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
