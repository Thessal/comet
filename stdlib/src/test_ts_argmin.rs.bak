#[cfg(test)]
mod tests {
    use crate::ts_argmin::TsArgminState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_argmin() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            5.0,  // w=[10, 20, 5], min is 5 at idx 2
            30.0, // w=[20, 5, 30], min is 5 at idx 1
            2.0,  // w=[5, 30, 2], min is 2 at idx 2
            f64::NAN, // w=[30, 2, NaN], min is 2 at idx 1
            2.0,  // w=[2, NaN, 2], min is 2 at idx 0 (tie goes to oldest)
        ];

        let mut state = TsArgminState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr());
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN,
            f64::NAN,
            2.0 / 3.0,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0 / 3.0,
            0.0 / 3.0,
        ];

        assert_eq!(our_output.len(), expected.len());
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if exp.is_nan() {
                assert!(ours.is_nan(), "Expected NaN at index {}", i);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-9, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
