#[cfg(test)]
mod tests {
    use crate::min::MinState;
    use crate::{BinaryOp, CometData};
    use crate::DataType as CometDataType;
    use polars::prelude::*;

    #[test]
    fn test_min_vs_polars() {
        let period = 1; 
        let len = 5;

        let input_data_a = vec![10.0, 20.0, 30.0, f64::NAN, 50.0];
        let input_data_b = vec![10.0, 25.0, 20.0, 40.0, 49.0];

        let mut state = MinState::new(period, len);
        let mut our_output = vec![0.0; len];
        
        let data_a = CometData {
            dtype: CometDataType::DataFrame,
            ptr: input_data_a.as_ptr(),
        };
        let data_b = CometData {
            dtype: CometDataType::DataFrame,
            ptr: input_data_b.as_ptr(),
        };

        state.step(data_a, data_b, our_output.as_mut_ptr(), len);

        let expected_out = vec![
            10.0,
            20.0,
            20.0,
            f64::NAN, // NaN propagates
            49.0
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
