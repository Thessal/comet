#[cfg(test)]
mod tests {
    use crate::r#const::ConstState;
    use crate::ConstOp;

    #[test]
    fn test_const() {
        let c = 42.0; 
        let len = 3;

        let mut state = ConstState::new(c, len);
        let mut out = vec![0.0; len];
        
        state.step(out.as_mut_ptr(), len);

        assert_eq!(out, vec![42.0, 42.0, 42.0]);
    }
}
