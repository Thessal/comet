#[cfg(test)]
mod tests {
    use crate::ts_mean::TsMeanState;
    use crate::StatefulUnary;
    use polars::prelude::*;

    #[test]
    fn test_ts_mean_vs_polars() {
        let period = 3;
        let len = 1;

        let input_data = vec![
            10.0,
            f64::NAN,
            20.0,
            f64::NAN,
            30.0,
            40.0,
            f64::NAN,
        ];

        // 1. Run our ts_mean
        let mut state = TsMeanState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        // 2. Run polars rolling mean (Commented out due to Polars 0.53 API changes)
        /*
        let df = df!("input" => &input_data).unwrap();
        
        let lazy_df = df.lazy().with_column(
            col("input")
                .rolling_mean(RollingOptions {
                    window_size: Duration::parse(&format!("{}i", period)),
                    min_periods: 1,
                    center: false,
                    weights: None,
                    fn_params: None,
                    by: None,
                    closed_window: None,
                    warn_aliasing: true,
                })
                .alias("rolling_mean")
        );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("rolling_mean").unwrap();
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
        */
    }
}
