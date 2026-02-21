#[cfg(test)]
mod tests {
    use crate::mid::MidState;
    use crate::BinaryOp;

    #[test]
    fn test_mid() {
        let period = 0; 
        let len = 4;

        let mut state = MidState::new(period, len);
        
        let a = vec![10.0, 20.0, f64::NAN, 100.0];
        let b = vec![30.0, 0.0, 50.0, f64::NAN];
        
        let mut out = vec![0.0; len];
        
        state.step(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), len);

        let expected = vec![
            20.0,
            10.0,
            f64::NAN,
            f64::NAN,
        ];
        
        for (i, (ours, exp)) in out.iter().zip(expected.iter()).enumerate() {
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
