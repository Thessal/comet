#[cfg(test)]
mod tests {
    use crate::abs::AbsState;
    use crate::UnaryOp;
    use polars::prelude::*;

    #[test]
    fn test_abs_vs_polars() {
        let period = 1; 
        let len = 5;

        let input_data = vec![
            10.0,
            -20.0,
            -25.5,
            f64::NAN,
            40.0,
        ];

        // 1. Run our abs
        let mut state = AbsState::new(period, len);
        let mut our_output = vec![0.0; len];
        state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: input_data.as_ptr() }, our_output.as_mut_ptr());

        // 2. Run polars standard abs
        let df = df!("input" => &input_data).unwrap();
        
        let lazy_df = df.lazy()
            .with_column(
                col("input").abs().alias("abs")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("abs").unwrap();
        
        let polars_out: Vec<Option<f64>> = polars_out_series.f64().unwrap().into_iter().collect();

        // 3. Compare outputs
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
