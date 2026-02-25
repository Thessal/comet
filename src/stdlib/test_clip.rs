#[cfg(test)]
mod tests {
    use crate::DataType as CometDataType;
    use crate::CometData;
    use crate::clip::comet_clip_step;
    use polars::prelude::*;

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

        let state = crate::clip::comet_clip_init(lower, upper, len);
        comet_clip_step(state, &data as *const CometData, our_output.as_mut_ptr(), len);
        crate::clip::comet_clip_free(state);

        let expected_out = vec![
            10.0,
            10.0,
            20.0,
            f64::NAN,
            25.0
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
