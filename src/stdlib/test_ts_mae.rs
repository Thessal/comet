#[cfg(test)]
mod tests {
    use crate::ts_mae::TsMaeState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_mae() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            30.0, // mean=20, mae=20/3 = 6.6666667
            40.0, // mean=30, mae=20/3
            f64::NAN, // mean=35, err=[5, 5], mae=5.0
            f64::NAN, // mean=40, err=[0], mae=0.0
            50.0, // mean=50, err=[0], mae=0.0
        ];

        let mut state = TsMaeState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN,
            f64::NAN,
            20.0 / 3.0,
            20.0 / 3.0,
            5.0,
            0.0,
            0.0,
        ];

        assert_eq!(our_output.len(), expected.len());
        // Verify math
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}, got {}", i, ours);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-7, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
