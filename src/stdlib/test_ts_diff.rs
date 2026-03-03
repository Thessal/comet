#[cfg(test)]
mod tests {
    use crate::ts_diff::TsDiffState;
    use crate::UnaryOp;
    use polars::prelude::*;
    use polars::series::ops::NullBehavior;

    #[test]
    fn test_ts_diff_vs_polars() {
        let period = 2; // Test a diff of 2 periods
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            25.0,
            f64::NAN,
            40.0,
            50.0,
        ];

        // 1. Run our ts_diff
        let mut state = TsDiffState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr());
            our_output.push(out[0]);
        }

        // 2. Run polars standard diff
        let df = df!("input" => &input_data).unwrap();
        
        let lazy_df = df.lazy()
            .with_column(
                col("input").diff((period as i32).into(), NullBehavior::Ignore).alias("diff")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("diff").unwrap();
        
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
