#[cfg(test)]
mod tests {
    use crate::ts_ffill::TsFfillState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_ffill() {
        let period = 2; // Maximum consecutive NaNs to fill is 2
        let len = 1;

        let input_data = vec![
            f64::NAN, // initial NaNs remain NaN
            10.0,     // valid
            f64::NAN, // filled (dist 1)
            f64::NAN, // filled (dist 2)
            f64::NAN, // not filled (dist 3 > period)
            f64::NAN, // not filled
            20.0,     // valid
            f64::NAN, // filled (dist 1)
        ];

        let mut state = TsFfillState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr());
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN,
            10.0,
            10.0,
            10.0,
            f64::NAN,
            f64::NAN,
            20.0,
            20.0,
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
