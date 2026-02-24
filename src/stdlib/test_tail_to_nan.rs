#[cfg(test)]
mod tests {
    use crate::DataType as CometDataType;
    use crate::CometData;
    use crate::tail_to_nan::comet_tail_to_nan_step;

    #[test]
    fn test_tail_to_nan_expected() {
        let len = 5;

        let input_data = vec![5.0, 10.0, 20.0, f64::NAN, 30.0];
        let lower = 10.0;
        let upper = 25.0;

        let mut our_output = vec![0.0; len];
        
        let data = CometData {
            dtype: CometDataType::DataFrame,
            ptr: input_data.as_ptr(),
        };

        comet_tail_to_nan_step(std::ptr::null_mut(), &data as *const CometData, lower, upper, our_output.as_mut_ptr(), len);

        let expected_out = vec![
            f64::NAN,
            10.0,
            20.0,
            f64::NAN,
            f64::NAN, // 30.0 > upper (25.0) -> NaN
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
