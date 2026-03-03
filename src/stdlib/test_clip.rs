#[cfg(test)]
mod tests {
    use crate::CometData;
    use crate::DataType as CometDataType;
    use crate::clip::ClipState;

    #[test]
    fn test_clip_vs_polars() {
        let len = 5;

        // FFI expects pointers
        let input_data = vec![5.0, -20.0, 20.0, f64::NAN, 30.0];
        let lower = 10.0;
        let upper = 25.0;

        let mut our_output = vec![0.0; len];

        let data = CometData {
            dtype: CometDataType::DataFrame,
            ptr: input_data.as_ptr(),
        };

        let mut state = ClipState::new(lower, upper, len);
        state.step(data, our_output.as_mut_ptr(), len);

        let expected_out = vec![10.0, 10.0, 20.0, f64::NAN, 25.0];

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
