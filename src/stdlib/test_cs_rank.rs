#[cfg(test)]
mod tests {
    use crate::cs_rank::CsRankState;
    use crate::UnaryOp;
    use polars::prelude::*;

    #[test]
    fn test_cs_rank_vs_polars() {
        let period = 10; 
        let len = 6;

        let input_data = vec![
            10.0,
            20.0,
            20.0,
            f64::NAN,
            40.0,
            50.0,
        ];

        // 1. Run our cs_rank
        let mut state = CsRankState::new(period, len);
        let mut our_output = vec![0.0; len];
        state.step(input_data.as_ptr(), our_output.as_mut_ptr(), len);

        // 2. Run polars standard rank
        // Polars: rank(method='average')
        let mut df = df!("input" => &input_data).unwrap();
        
        let lazy_df = df.lazy()
            .with_column(
                when(col("input").is_nan())
                .then(lit(NULL))
                .otherwise(col("input"))
                .alias("input_clean")
            )
            .with_column(
                col("input_clean").rank(
                    RankOptions {
                        method: RankMethod::Average,
                        descending: false
                    },
                    None
                ).alias("rank")
            );

        let out_df = lazy_df.collect().unwrap();
        let polars_out_series = out_df.column("rank").unwrap();
        
        // Polars rank might return f32 or f64 or u32 depending on version/method.
        // Try casting to f64 to be safe.
        let casted = polars_out_series.cast(&DataType::Float64).unwrap();
        let polars_out: Vec<Option<f64>> = casted.f64().unwrap().into_iter().collect();

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
