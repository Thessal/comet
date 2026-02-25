#[cfg(test)]
mod tests {
    use crate::ts_decay_exp::TsDecayExpState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_decay_exp() {
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

        let mut state = TsDecayExpState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(crate::CometData { dtype: crate::DataType::DataFrame, ptr: &[*val] as *const f64 }, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        // Expected manually calculated:
        // alpha = 2 / 4 = 0.5
        // i=0: 1.0 (init)
        // i=1: 2.0*0.5 + 1.0*0.5 = 1.5
        // i=2: 3.0*0.5 + 1.5*0.5 = 2.25
        // i=3: 4.0*0.5 + 2.25*0.5 = 3.125
        // i=4: 5.0*0.5 + 3.125*0.5 = 4.0625
        // i=5: NaN -> ignored, keeps 4.0625.
        // i=6: 7.0*0.5 + 4.0625*0.5 = 5.53125
        
        let expected = vec![
            1.0,
            1.5,
            2.25,
            3.125,
            4.0625,
            4.0625,
            5.53125,
        ];

        assert_eq!(our_output.len(), expected.len());
        for (i, (ours, exp)) in our_output.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}", i);
            } else {
                assert!(
                    (ours - exp).abs() < 1e-9, 
                    "Mismatch at {}: ours={}, exp={}", i, ours, exp
                );
            }
        }
    }
}
