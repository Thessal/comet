#[cfg(test)]
mod tests {
    use crate::CometData;
    use crate::DataType as CometDataType;
    use crate::TernaryOp;

    #[test]
    fn test_tradewhen_expected() {
        let period = 2; // Ffill limit
        let len = 1;

        let signal_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let enter_data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let exit_data = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];

        let mut state = crate::tradewhen::TradewhenState::new(period, len);
        let mut our_output = Vec::new();

        for i in 0..signal_data.len() {
            let c_sig = CometData {
                dtype: CometDataType::DataFrame,
                ptr: &signal_data[i] as *const f64,
            };
            let c_ent = CometData {
                dtype: CometDataType::DataFrame,
                ptr: &enter_data[i] as *const f64,
            };
            let c_ext = CometData {
                dtype: CometDataType::DataFrame,
                ptr: &exit_data[i] as *const f64,
            };

            let mut out = vec![0.0; len];
            state.step(c_sig, c_ent, c_ext, out.as_mut_ptr(), len);
            our_output.push(out[0]);
        }

        let expected_out = vec![
            f64::NAN, // enter 0, exit 0 -> None
            20.0,     // enter 1
            20.0,     // ffill 1
            f64::NAN, // exit 1 -> NaN
            f64::NAN, // ffill ends or NaN sustained
            60.0,     // enter 1
            60.0,     // ffill 1
        ];

        assert_eq!(our_output.len(), expected_out.len());

        for (i, (ours, expected)) in our_output.iter().zip(expected_out.iter()).enumerate() {
            if f64::is_nan(*ours) && f64::is_nan(*expected) {
                continue;
            }
            assert!(
                (ours - expected).abs() < 1e-9,
                "Mismatch at {}: ours={}, expected={}",
                i,
                ours,
                expected
            );
        }
    }
}
