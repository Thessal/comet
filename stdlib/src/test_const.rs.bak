#[cfg(test)]
mod tests {
    use crate::ZeroAryOp;
    use crate::r#const::ConstState;

    #[test]
    fn test_const() {
        let c = 42.0;
        let len = 3;

        let mut state = ConstState::new(c, len);
        let mut out = vec![0.0; 1];

        state.step(out.as_mut_ptr());

        assert_eq!(out, vec![42.0]);
    }
}
