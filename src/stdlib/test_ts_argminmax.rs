#[cfg(test)]
mod tests {
    use crate::ts_argminmax::TsArgminmaxState;
    use crate::UnaryOp;

    #[test]
    fn test_ts_argminmax() {
        let period = 3; 
        let len = 1;

        let input_data = vec![
            10.0,
            20.0,
            5.0,  // w=[10, 20, 5], min is 5 at idx 2, max is 20 at idx 1. Diff: (2 - 1)/3 = 1/3
            30.0, // w=[20, 5, 30], min is 5 at idx 1, max is 30 at idx 2. Diff: (1 - 2)/3 = -1/3
            2.0,  // w=[5, 30, 2], min is 2 at idx 2, max is 30 at idx 1. Diff: (2 - 1)/3 = 1/3
            f64::NAN, // w=[30, 2, NaN], min is 2 at idx 1, max is 30 at idx 0. Diff: (1 - 0)/3 = 1/3
            2.0,  // w=[2, NaN, 2], min is 2 at idx 0, max is 2 at idx 0. Diff: 0/3 = 0
        ];

        let mut state = TsArgminmaxState::new(period, len);
        let mut our_output = Vec::new();

        for val in &input_data {
            let mut out = vec![0.0; len];
            state.step(&[*val] as *const f64, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected = vec![
            f64::NAN,
            f64::NAN,
            1.0 / 3.0,
            -1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.0 / 3.0,
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
