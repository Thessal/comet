#[cfg(test)]
mod tests {
    use crate::UnaryOp;
    use crate::ts_std::TsStdState;

    #[test]
    fn test_ts_std() {
        let period = 3;
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            30.0,     // w=[10, 20, 30] -> mean=20, std=sqrt(200/3) = 8.1649658
            40.0,     // w=[20, 30, 40] -> std=8.1649658
            f64::NAN, // w=[30, 40, NaN] -> valid=[30, 40], mean=35, std=sqrt(((30-35)^2 + (40-35)^2)/2) = 5.0
            f64::NAN, // w=[40, NaN, NaN] -> valid=[40], mean=40, std=0.0
            50.0,     // w=[NaN, NaN, 50] -> std=0.0
        ];

        let mut state = TsStdState::new(period, len);
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

        let expected = vec![
            f64::NAN,
            f64::NAN,
            (200.0f64 / 3.0).sqrt(),
            (200.0f64 / 3.0).sqrt(),
            5.0,
            0.0,
            0.0,
        ];

        assert_eq!(our_output.len(), expected.len());
        // Verify math
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(
                    f64::is_nan(*ours),
                    "Expected NaN at index {}, got {}",
                    i,
                    ours
                );
            } else {
                assert!(
                    (ours - exp).abs() < 1e-7,
                    "Mismatch at {}: ours={}, exp={}",
                    i,
                    ours,
                    exp
                );
            }
        }
    }
}
