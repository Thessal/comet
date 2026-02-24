#[cfg(test)]
mod tests {
    use crate::power::PowerState;
    use crate::{BinaryOp, CometData};
    use crate::DataType as CometDataType;
    use polars::prelude::*;

    #[test]
    fn test_power_vs_polars() {
        let period = 1; 
        let len = 5;

        let input_data_a = vec![2.0, 3.0, 10.0, f64::NAN, 5.0];
        let input_data_b = vec![3.0, 2.0, 0.5, 4.0, 0.0];

        let mut state = PowerState::new(period, len);
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

        let df = df!(
            "a" => &input_data_a,
            "b" => &input_data_b
        ).unwrap();
        
        let lazy_df = df.lazy()
            .with_column(
                col("a").pow(col("b")).alias("out")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("out").unwrap();
        
        let polars_out: Vec<Option<f64>> = polars_out_series.f64().unwrap().into_iter().collect();

        assert_eq!(our_output.len(), polars_out.len());
        
        for (i, (ours, theirs)) in our_output.iter().zip(polars_out.iter()).enumerate() {
            let theirs_val = theirs.unwrap_or(f64::NAN);
            if f64::is_nan(*ours) && f64::is_nan(theirs_val) {
                continue;
            }
            assert!(
                (ours - theirs_val).abs() < 1e-9, 
                "Mismatch at {}: ours={}, theirs={}", i, ours, theirs_val
            );
        }
    }
}
