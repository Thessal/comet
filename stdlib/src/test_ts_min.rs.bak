#[cfg(test)]
mod tests {
    use crate::UnaryOp;
    use crate::ts_min::TsMinState;

    #[test]
    fn test_ts_min_expected() {
        let period = 3;
        let len = 1;

        let input_data = vec![
            20.0,
            10.0,
            15.0,     // min: 10.0
            f64::NAN, // min: 10.0 (out of 10, 15, NaN)
            5.0,      // min: 5.0 (out of 15, NaN, 5)
            30.0,     // min: 5.0 (out of NaN, 5, 30)
            40.0,     // min: 5.0 (out of 5, 30, 40)
        ];

        let mut state = TsMinState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(
                crate::CometData {
                    dtype: crate::DataType::DataFrame,
                    ptr: &[*val] as *const f64,
                },
                out.as_mut_ptr(),
            );
            our_output.push(out[0]);
        }

        let expected_out = vec![
            20.0, // NOTE: assumes min_period = 1
            10.0, 10.0, 10.0, 5.0, 5.0, 5.0,
        ];

        assert_eq!(our_output.len(), expected_out.len());

        for (i, (ours, expected)) in our_output.iter().zip(expected_out.iter()).enumerate() {
            if f64::is_nan(*ours) && f64::is_nan(*expected) {
                continue;
            }
            assert!(
                (ours - expected).abs() < 1e-9,
                "Mismatch at {}: ours={}, expected={}",
                i,
                ours,
                expected
            );
        }
    }
}
