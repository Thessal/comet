#[cfg(test)]
mod tests {
    use crate::ts_decay_linear::TsDecayLinearState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_decay_linear() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            7.0,
        ];

        let mut state = TsDecayLinearState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        // Expected manually calculated:
        // sum_weights = 6.0
        // i=0: 1.0 (len < 3) -> NaN
        // i=1: 2.0 (len < 3) -> NaN
        // i=2: 3 -> (1*1 + 2*2 + 3*3)/6 = 14/6 = 2.3333333333333335
        // i=3: 4 -> (1*2 + 2*3 + 3*4)/6 = 20/6 = 3.3333333333333335
        // i=4: 5 -> (1*3 + 2*4 + 3*5)/6 = 26/6 = 4.333333333333333
        // i=5: NaN -> (1*4 + 2*5 + 3*NaN)/6 = NaN
        // i=6: 7 -> (1*5 + 2*NaN + 3*7)/6 = NaN
        
        let expected = vec![
            f64::NAN,
            f64::NAN,
            14.0 / 6.0,
            20.0 / 6.0,
            26.0 / 6.0,
            f64::NAN,
            f64::NAN,
        ];

        assert_eq!(our_output.len(), expected.len());
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}, got {}", i, ours);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-9, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
