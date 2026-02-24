#[cfg(test)]
mod tests {
    use crate::DataType as CometDataType;
    use crate::CometData;
    use crate::r#where::comet_where_step;
    use polars::prelude::*;

    #[test]
    fn test_where_vs_polars() {
        let len = 5;

        let cond_data = vec![1.0, -1.0, 1.0, f64::NAN, 0.0];
        let t_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let f_data = vec![100.0, 200.0, 300.0, 400.0, 500.0];

        let mut our_output = vec![0.0; len];
        
        let c_obj = CometData { dtype: CometDataType::DataFrame, ptr: cond_data.as_ptr() };
        let t_obj = CometData { dtype: CometDataType::DataFrame, ptr: t_data.as_ptr() };
        let f_obj = CometData { dtype: CometDataType::DataFrame, ptr: f_data.as_ptr() };

        comet_where_step(
            std::ptr::null_mut(),
            &c_obj as *const CometData,
            &t_obj as *const CometData,
            &f_obj as *const CometData,
            our_output.as_mut_ptr(),
            len
        );

        let expected_out = vec![
            10.0,
            200.0,
            30.0,
            f64::NAN,
            f64::NAN,
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
