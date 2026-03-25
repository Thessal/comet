#[cfg(test)]
mod tests {
    use crate::CometData;
    use crate::DataType as CometDataType;
    use crate::MatrixOp;
    use crate::covariance::CovarianceState;

    #[test]
    fn test_covariance_expected() {
        let period = 3;
        let len = 2; // Testing with N=2 cross-sections

        // Columns representing 2 assets over 4 periods
        let r_data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, f64::NAN],
        ];

        let mut state = CovarianceState::new(period, len);
        let mut our_outputs = Vec::new();

        for row in r_data.iter() {
            let c_sig = CometData {
                dtype: CometDataType::DataFrame,
                ptr: row.as_ptr(),
            };
            let mut out = vec![0.0; len * len];
            state.step(c_sig, out.as_mut_ptr());
            our_outputs.push(out);
        }

        // period=3, so first 2 periods are NaN.
        // 3rd period cov between X=[1,2,3], Y=[2,4,6]
        // cov(X, X) = 1, cov(Y, Y) = 4, cov(X, Y) = 2

        let expected_3rd = vec![1.0, 2.0, 2.0, 4.0];

        assert_eq!(our_outputs.len(), 4);

        for (i, ours) in our_outputs.iter().enumerate() {
            if i < period - 1 {
                assert!(f64::is_nan(ours[0]));
            } else if i == 2 {
                for (o, e) in ours.iter().zip(expected_3rd.iter()) {
                    assert!(
                        (o - e).abs() < 1e-9,
                        "Mismatch at {}: ours={}, expected={}",
                        i,
                        o,
                        e
                    );
                }
            }
        }
    }
}
