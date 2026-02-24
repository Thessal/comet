#[cfg(test)]
mod tests {
    use crate::ts_max::TsMaxState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_max_expected() {
        let period = 3;
        let len = 1;

        let input_data = vec![
            0.0,      
            5.0,      
            20.0,      // max: 20.0 (out of 10, 5, 20)
            f64::NAN,  // max: 20.0 (out of 5, 20, NaN)
            15.0,      // max: 20.0 (out of 20, NaN, 15)
            5.0,       // max: 15.0 (out of NaN, 15, 5)
            40.0,      // max: 40.0 (out of 15, 5, 40)
        ];

        let mut state = TsMaxState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected_out = vec![
            0.0,
            5.0, // min_periods = 1
            20.0,
            20.0,
            20.0,
            15.0,
            40.0,
        ];

        assert_eq!(our_output.len(), expected_out.len());
        
        for (i, (ours, expected)) in our_output.iter().zip(expected_out.iter()).enumerate() {
            if f64::is_nan(*ours) && f64::is_nan(*expected) {
                continue;
            }
            assert!(
                (ours - expected).abs() < 1e-9, 
                "Mismatch at {}: ours={}, expected={}", i, ours, expected
            );
        }
    }
}
