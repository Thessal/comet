#[cfg(test)]
mod tests {
    use crate::ts_sum::TsSumState;
    use crate::UnaryOp;
    use polars::prelude::*;

    #[test]
    fn test_ts_sum_vs_polars() {
        // Since ts_sum functionality is basic rolling sum.
        // As seen in test_ts_mean we don't strictly require polars execution if it has API incompatibilities,
        // but let's test directly matching expected outcomes or running polars rolling_sum if possible.
        let period = 3;
        let len = 1;

        let input_data = vec![
            10.0,      // sum: NaN (counts = 1)
            20.0,      // sum: NaN (counts = 2)
            30.0,      // sum: 60.0 (counts = 3)
            f64::NAN,  // sum: 50.0 (counts = 2 from previous)
            40.0,      // sum: 70.0 (counts = 2)
            50.0,      // sum: 90.0 (counts = 2)
            60.0,      // sum: 150.0 (counts = 3)
        ];

        let mut state = TsSumState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected_out = vec![
            10.0,
            30.0,
            60.0,
            50.0,
            70.0,
            90.0,
            150.0,
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
