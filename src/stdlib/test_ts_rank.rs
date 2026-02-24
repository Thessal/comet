#[cfg(test)]
mod tests {
    use crate::ts_rank::TsRankState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_rank_expected() {
        let period = 3;
        let len = 1;

        let input_data = vec![
            10.0,      
            5.0,      
            20.0,      // rank of 20 in (10, 5, 20) -> sorted: (5, 10, 20) -> rank 3.0
            f64::NAN,  // rank of NaN is NaN
            15.0,      // rank of 15 in (20, NaN, 15) -> sorted: (15, 20) -> rank 1.0
            15.0,      // rank of 15 in (NaN, 15, 15) -> sorted: (15, 15) -> rank 1.5
            15.0,      // rank of 15 in (15, 15, 15) -> sorted: (15, 15, 15) -> rank 2.0
        ];

        let mut state = TsRankState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected_out = vec![
            f64::NAN, // N=1
            0.0,      // N=2, rank=1, (1-1)/(2-1)
            1.0,      // N=3, rank=3, (3-1)/(3-1)
            f64::NAN,
            0.0,      // N=2, rank=1, (1-1)/(2-1)
            0.5,      // N=2, rank=1.5, (1.5-1)/(2-1)
            0.5,      // N=3, rank=2, (2-1)/(3-1)
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
