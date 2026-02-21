#[cfg(test)]
mod tests {
    use crate::ts_delay::TsDelayState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_delay() {
        let period = 2; // With period 2, it needs 2 items in history before shifting
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            30.0, // Should output 10.0
            40.0, // Should output 20.0
            f64::NAN, // Should output 30.0
            50.0, // Should output NaN
            60.0, // Should output 50.0
        ];

        let mut state = TsDelayState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN, 
            f64::NAN,
            10.0,
            20.0,
            30.0,
            40.0,
            f64::NAN,
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
