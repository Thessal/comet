#[cfg(test)]
mod tests {
    use crate::ts_zscore::TsZscoreState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_zscore() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            10.0,
            30.0,
            20.0, // mean=20, std=sqrt(200/3)=8.16496. z=(20-20)/std = 0
            50.0, // w=[30, 20, 50], mean=33.33. z=(50-33.33)/std
            f64::NAN, // w=[20, 50, NaN], cur=NaN. z=NaN
            f64::NAN, // w=[50, NaN, NaN], cur=NaN. z=NaN
            100.0, // w=[NaN, NaN, 100], std=0. z=NaN
        ];

        let mut state = TsZscoreState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected_std2 = (((30.0f64 - 100.0/3.0).powi(2) + (20.0f64 - 100.0/3.0).powi(2) + (50.0f64 - 100.0/3.0).powi(2))/3.0).sqrt();

        let expected = vec![
            f64::NAN,
            f64::NAN,
            0.0,
            (50.0 - 100.0/3.0) / expected_std2,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];

        assert_eq!(our_output.len(), expected.len());
        // Verify math
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}", i);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-7, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
