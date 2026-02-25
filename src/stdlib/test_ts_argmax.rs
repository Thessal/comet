#[cfg(test)]
mod tests {
    use crate::ts_argmax::TsArgmaxState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_argmax() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            5.0,  // w=[10, 20, 5], max is 20 at idx 1
            30.0, // w=[20, 5, 30], max is 30 at idx 2
            2.0,  // w=[5, 30, 2], max is 30 at idx 1
            f64::NAN, // w=[30, 2, NaN], max is 30 at idx 0
            2.0,  // w=[2, NaN, 2], max is 2 at idx 0 (tie goes to oldest)
        ];

        let mut state = TsArgmaxState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN,
            f64::NAN,
            1.0 / 3.0,
            2.0 / 3.0,
            1.0 / 3.0,
            0.0 / 3.0,
            0.0 / 3.0,
        ];

        assert_eq!(our_output.len(), expected.len());
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}", i);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-9, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
