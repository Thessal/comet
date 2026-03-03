#[cfg(test)]
mod tests {
    use crate::mid::MidState;
    use crate::{BinaryOp, CometData, DataType};

    #[test]
    fn test_mid_df_df() {
        let period = 0; 
        let len = 4;
        let mut state = MidState::new(period, len);
        
        let a_vec = vec![10.0, 20.0, f64::NAN, 100.0];
        let b_vec = vec![30.0, 0.0, 50.0, f64::NAN];
        
        let a = CometData { dtype: DataType::DataFrame, ptr: a_vec.as_ptr() };
        let b = CometData { dtype: DataType::DataFrame, ptr: b_vec.as_ptr() };
        
        let mut out = vec![0.0; len];
        state.step(a, b, out.as_mut_ptr());

        let expected = vec![20.0, 10.0, f64::NAN, f64::NAN];
        
        for (i, (ours, exp)) in out.iter().zip(expected.iter()).enumerate() {
            if f64::is_nan(*exp) {
                assert!(f64::is_nan(*ours), "Expected NaN at index {}", i);
            } else {
                assert!((ours - exp).abs() < 1e-9);
            }
        }
    }
}
