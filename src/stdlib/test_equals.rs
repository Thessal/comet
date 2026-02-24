#[cfg(test)]
mod tests {
    use crate::equals::EqualsState;
    use crate::{BinaryOp, CometData};
    use crate::DataType as CometDataType;
    use polars::prelude::*;

    #[test]
    fn test_equals_vs_polars() {
        let period = 1; 
        let len = 5;

        let input_data_a = vec![10.0, 20.0, 30.0, f64::NAN, 50.0];
        let input_data_b = vec![10.0, 25.0, 30.0, 40.0, 50.0];

        let mut state = EqualsState::new(period, len);
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
                col("a").eq(col("b")).cast(polars::datatypes::DataType::Float64).alias("out")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("out").unwrap();
        
        let polars_out: Vec<Option<f64>> = polars_out_series.f64().unwrap().into_iter().collect();

        assert_eq!(our_output.len(), polars_out.len());
        
        for (i, (ours, theirs)) in our_output.iter().zip(polars_out.iter()).enumerate() {
            let theirs_val = theirs.unwrap_or(0.0);
            
            // NaN handling for equals usually is false (0.0), so let's verify correctly
            // If polars turns NaN == NaN -> null, it comes out as null/None. Let's safely coalesce
            
            assert!(
                (ours - theirs_val).abs() < 1e-9, 
                "Mismatch at {}: ours={}, theirs={}", i, ours, theirs_val
            );
        }
    }
}
