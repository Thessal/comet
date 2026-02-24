#[cfg(test)]
mod tests {
    use crate::cs_zscore::CsZscoreState;
    use crate::UnaryOp;
    use polars::prelude::*;

    #[test]
    fn test_cs_zscore_vs_polars() {
        let period = 10; // cs_zscore doesn't actually use period since it's Cross-Sectional, but required by API
        let len = 5;

        // Cross Section Data for a single time step
        // We will perform one step where the length of the array is 5.
        // NaN should be ignored in the calculation.
        let input_data = vec![
            10.0,
            20.0,
            f64::NAN,
            40.0,
            50.0,
        ];

        // 1. Run our cs_zscore
        let mut state = CsZscoreState::new(period, len);
        let mut our_output = vec![0.0; len];
        state.step(input_data.as_ptr(), our_output.as_mut_ptr(), len);

        // Polars: (val - mean) / std_dev
        // In order to only use non-NaN values for mean/std, Polars handles this correctly
        // However, we want the Z-score column to be evaluated row-by-row on valid entries.
        let df = df!("input" => &input_data).unwrap();
        
        // Convert f64::NAN to Nulls so polars ignores them correctly in aggregation
        let lazy_df = df.lazy()
            .with_column(
                when(col("input").is_nan())
                .then(lit(NULL))
                .otherwise(col("input"))
                .alias("input_clean")
            )
            .with_column(
                ((col("input_clean") - col("input_clean").mean()) / col("input_clean").std(1))
                    .alias("zscore")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("zscore").unwrap();
        let polars_out: Vec<Option<f64>> = polars_out_series.f64().unwrap().into_iter().collect();

        // 3. Compare outputs
        assert_eq!(our_output.len(), polars_out.len());
        
        for (i, (ours, theirs)) in our_output.iter().zip(polars_out.iter()).enumerate() {
            let theirs_val = theirs.unwrap_or(f64::NAN);
            if ours.is_nan() && theirs_val.is_nan() {
                continue;
            }
            // Use an epsilon for floating point comparison
            assert!(
                (ours - theirs_val).abs() < 1e-9, 
                "Mismatch at {}: ours={}, theirs={}", i, ours, theirs_val
            );
        }
    }
}
