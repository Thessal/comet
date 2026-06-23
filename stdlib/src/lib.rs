// Minimal version of stdlib
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]
mod op_add;
mod op_cross_section;
mod op_cs_zscore;
mod op_data;
mod op_divide;
mod op_flip;
mod op_multiply;
mod op_pow;
mod op_subtract;
mod op_time_series;
mod op_ts_mean;

pub mod types;
use tch::Kind;
use types::Signal;

pub struct OperatorSpec {
    pub name: &'static str,
    pub inputs: &'static [Signal],
    pub output_shape: Signal,
    pub execute: fn(&[Signal]) -> Signal,
}

impl std::fmt::Debug for OperatorSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OperatorSpec {{ name: {} }}", self.name)
    }
}

impl PartialEq for OperatorSpec {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl From<&str> for &OperatorSpec {
    fn from(sig_name: &str) -> Self {
        match sig_name {
            "data" => &op_data::OP_DATA,
            "add" => &op_add::OP_ADD,
            "subtract" => &op_subtract::OP_SUBTRACT,
            "divide" => &op_divide::OP_DIVIDE,
            "multiply" => &op_multiply::OP_MULTIPLY,
            "flip" => &op_flip::OP_FLIP,
            "pow" => &op_pow::OP_POW,
            // "cs_rank" => &op_cs_rank::OP_CS_RANK, same with rank()
            "cs_zscore" => &op_cs_zscore::OP_CS_ZSCORE,
            "ts_mean" => &op_ts_mean::OP_TS_MEAN,
            "rank" => &op_cross_section::OP_RANK,
            "rank_add" => &op_cross_section::OP_RANK_ADD,
            "rank_sub" => &op_cross_section::OP_RANK_SUB,
            "rank_mul" => &op_cross_section::OP_RANK_MUL,
            "rank_div" => &op_cross_section::OP_RANK_DIV,
            "sign" => &op_cross_section::OP_SIGN,
            "sigmoid" => &op_cross_section::OP_SIGMOID,
            "delay" => &op_time_series::OP_DELAY,
            "delta" => &op_time_series::OP_DELTA,
            "ts_return" => &op_time_series::OP_TS_RETURN,
            "ts_max" => &op_time_series::OP_TS_MAX,
            "ts_min" => &op_time_series::OP_TS_MIN,
            "ts_argmax" => &op_time_series::OP_TS_ARGMAX,
            "ts_argmin" => &op_time_series::OP_TS_ARGMIN,
            "ts_sum" => &op_time_series::OP_TS_SUM,
            "ts_prod" => &op_time_series::OP_TS_PROD,
            "ts_nanmean" => &op_time_series::OP_TS_NANMEAN,
            "ts_stddev" => &op_time_series::OP_TS_STDDEV,
            "ts_rank" => &op_time_series::OP_TS_RANK,
            "ts_cov" => &op_time_series::OP_TS_COV,
            "ts_corr" => &op_time_series::OP_TS_CORR,
            // "ts_diff" => OperatorSpec {
            //     name: "ts_diff",
            //     inputs: &[Signal::Int(None), Signal::DataFrame(None)],
            //     output_shape: Signal::DataFrame(None),
            //     execute: |args| match (&args[0], &args[1]) {
            //         (Signal::Int(Some(period)), Signal::DataFrame(Some(a))) => {
            //             let p = *period as i64;
            //             let shifted = a.roll(&[p], &[0]);
            //             if p > 0 && p <= a.size()[0] {
            //                 let mut slice = shifted.narrow(0, 0, p);
            //                 let nan = tch::Tensor::full(&[1], f64::NAN, (a.kind(), a.device()));
            //                 let _ = slice.copy_(&nan);
            //             }
            //             Signal::DataFrame(Some(a - shifted))
            //         }
            //         _ => panic!("ts_diff expected Int and DataFrame"),
            //     },
            // },
            // "ts_mean" => OperatorSpec {
            //     name: "ts_mean",
            //     inputs: &[Signal::DataFrame(None), Signal::Int(None)],
            //     output_shape: Signal::DataFrame(None),
            //     execute: |args| {
            //         match (&args[0], &args[1]) {
            //             (Signal::DataFrame(Some(a)), Signal::Int(Some(_period))) => {
            //                 // Dummy implementation for now to pass tests (returns same DF)
            //                 // TODO: implement rolling window mean with unfold/conv1d
            //                 Signal::DataFrame(Some(a.shallow_clone()))
            //             }
            //             _ => panic!("ts_mean expected DataFrame and Int"),
            //         }
            //     },
            // },
            // "ts_rank" => OperatorSpec {
            //     name: "ts_rank",
            //     inputs: &[Signal::DataFrame(None), Signal::Int(None)],
            //     output_shape: Signal::DataFrame(None),
            //     execute: |args| {
            //         match (&args[0], &args[1]) {
            //             (Signal::DataFrame(Some(a)), Signal::Int(Some(_period))) => {
            //                 // Dummy implementation for now to pass tests (returns same DF)
            //                 // TODO: implement rolling window rank
            //                 Signal::DataFrame(Some(a.shallow_clone()))
            //             }
            //             _ => panic!("ts_rank expected DataFrame and Int"),
            //         }
            //     },
            // },
            // "consume_float" => OperatorSpec {
            //     name: "consume_float",
            //     inputs: &[Signal::Float(None)],
            //     output_shape: Signal::Void,
            //     execute: |_args| Signal::Void,
            // },
            _ => panic!("Could not find {} in the stdlib", sig_name),
        }
    }
}

impl OperatorSpec {
    pub fn execute(&self, args: &[Signal]) -> Result<Signal, String> {
        let arity = self.inputs.len();
        if args.len() < arity {
            return Err(format!("Stack underflow for {}", self.name));
        }

        for (arg, expected) in args.iter().zip(self.inputs.iter()) {
            if std::mem::discriminant(arg) != std::mem::discriminant(expected) {
                return Err(format!(
                    "Type mistmatch for {}, arg: {:?}, expected: {:?}",
                    self.name, arg, expected
                ));
            }
        }

        Ok((self.execute)(args))
    }
}

// TODO:
// 4. numerical accuracy neeed to be checked.

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    fn generate_args(spec: &OperatorSpec, device: tch::Device) -> Vec<Signal> {
        spec.inputs
            .iter()
            .map(|s| {
                match s {
                    Signal::DataFrame(_) => {
                        // Random array with values in [0, 1)
                        Signal::DataFrame(Some(tch::Tensor::rand(
                            &crate::types::SIZE,
                            (tch::Kind::Float, device),
                        )))
                    }
                    Signal::Int(_) => {
                        Signal::Int(Some(5)) // rolling window of 5
                    }
                    Signal::Float(_) => Signal::Float(Some(0.5)),
                    _ => panic!("Unsupported signal type for tests"),
                }
            })
            .collect()
    }

    fn get_all_ops() -> Vec<&'static str> {
        vec![
            "data",
            "add",
            "subtract",
            "divide",
            "multiply",
            "flip",
            "pow",
            // "cs_rank",
            "cs_zscore",
            "ts_mean",
            "rank",
            "rank_add",
            "rank_sub",
            "rank_mul",
            "rank_div",
            "sign",
            "sigmoid",
            "delay",
            "delta",
            "ts_return",
            "ts_max",
            "ts_min",
            "ts_argmax",
            "ts_argmin",
            "ts_sum",
            "ts_prod",
            "ts_nanmean",
            "ts_stddev",
            "ts_rank",
            "ts_cov",
            "ts_corr",
        ]
    }

    #[test]
    fn test_data_frame_size() {
        let device = tch::Device::Cpu;
        let ops = get_all_ops();
        for op_name in ops {
            if op_name == "data" {
                continue;
            }
            let spec: &OperatorSpec = op_name.into();
            let args = generate_args(spec, device);
            let out = spec.execute(&args).unwrap();
            if let Signal::DataFrame(Some(t)) = out {
                assert_eq!(
                    t.size(),
                    crate::types::SIZE,
                    "Operator {} changed dataframe size",
                    op_name
                );
            }
        }
    }

    #[test]
    fn test_cross_sectional_shuffle() {
        let device = tch::Device::Cpu;
        let ops = get_all_ops();
        for op_name in ops {
            if op_name == "data" {
                continue;
            }
            let spec: &OperatorSpec = op_name.into();
            let args = generate_args(spec, device);

            let num_assets = crate::types::SIZE[1];
            if num_assets < 2 {
                continue;
            }

            // Create a permutation of the columns: reverse them for simplicity
            let perm_vec: Vec<i64> = (0..num_assets).rev().collect();
            let perm = tch::Tensor::from_slice(&perm_vec).to_device(device);

            let mut shuffled_args = Vec::new();
            for arg in &args {
                match arg {
                    Signal::DataFrame(Some(t)) => {
                        shuffled_args.push(Signal::DataFrame(Some(t.index_select(1, &perm))));
                    }
                    _ => shuffled_args.push(arg.clone()),
                }
            }

            let out_orig = spec.execute(&args).unwrap();
            let out_shuffled = spec.execute(&shuffled_args).unwrap();

            if let (Signal::DataFrame(Some(t_orig)), Signal::DataFrame(Some(t_shuf))) =
                (out_orig, out_shuffled)
            {
                let t_orig_shuffled = t_orig.index_select(1, &perm);
                let is_close = t_orig_shuffled.isclose(&t_shuf, 1e-4, 1e-6, true);
                let max_diff = (&t_orig_shuffled - &t_shuf).abs().max().double_value(&[]);
                let is_all_true = i64::try_from(is_close.all()).unwrap() != 0;
                assert!(
                    is_all_true,
                    "Operator {} failed cross-sectional shuffle test, max diff: {}",
                    op_name, max_diff
                );
            }
        }
    }

    #[test]
    fn test_forward_looking() {
        let device = tch::Device::Cpu;
        let ops = get_all_ops();
        let t_split = std::cmp::min(100, crate::types::SIZE[0] / 2);

        for op_name in ops {
            if op_name == "data" {
                continue;
            }
            let spec: &OperatorSpec = op_name.into();
            let args = generate_args(spec, device);

            let out_orig = spec.execute(&args).unwrap();

            let mut future_args = Vec::new();
            for arg in &args {
                match arg {
                    Signal::DataFrame(Some(t)) => {
                        let mut modified = t.shallow_clone(); // + 0.0; // force a deep copy by adding 0.0
                        // Mutate the "future" data (after t_split)
                        let mut future_slice =
                            modified.narrow(0, t_split, crate::types::SIZE[0] - t_split);
                        let _ = future_slice.copy_(&tch::Tensor::rand(
                            future_slice.size().as_slice(),
                            (tch::Kind::Float, device),
                        ));
                        future_args.push(Signal::DataFrame(Some(modified)));
                    }
                    _ => future_args.push(arg.clone()),
                }
            }

            let out_future = spec.execute(&future_args).unwrap();

            if let (Signal::DataFrame(Some(t_orig)), Signal::DataFrame(Some(t_fut))) =
                (out_orig, out_future)
            {
                let t_orig_past = t_orig.narrow(0, 0, t_split);
                let t_fut_past = t_fut.narrow(0, 0, t_split);

                let is_close = t_orig_past.isclose(&t_fut_past, 1e-5, 1e-8, true);
                let is_all_true = i64::try_from(is_close.all()).unwrap() != 0;
                assert!(
                    is_all_true,
                    "Operator {} failed forward-looking test",
                    op_name
                );
            }
        }
    }

    #[test]
    fn test_nan_instrument_invariance() {
        // It means that the operator result is invariant under adding cross-sectional addition of nan values, except the result contains additional nan value.
        // In other word, cross-sectional trimming of nan values don't change the result.
        let device = tch::Device::Cpu;
        let ops = get_all_ops();

        for op_name in ops {
            if op_name == "data" {
                continue;
            }
            // Temporarily ignore cs_rank if it's commented out in lib.rs
            if op_name == "cs_rank" {
                continue;
            }
            let spec: &OperatorSpec = op_name.into();

            let args_base = generate_args(spec, device);
            let out_base = spec.execute(&args_base).unwrap();

            let num_nan_cols = 10;
            let mut args_nan = Vec::new();

            for arg in &args_base {
                match arg {
                    Signal::DataFrame(Some(t)) => {
                        let rows = t.size()[0];
                        let nan_cols = tch::Tensor::full(
                            &[rows, num_nan_cols],
                            f64::NAN,
                            (tch::Kind::Float, device),
                        );
                        let t_with_nan = tch::Tensor::cat(&[t, &nan_cols], 1);
                        args_nan.push(Signal::DataFrame(Some(t_with_nan)));
                    }
                    _ => args_nan.push(arg.clone()),
                }
            }

            let out_nan = spec.execute(&args_nan).unwrap();

            if let (Signal::DataFrame(Some(t_orig)), Signal::DataFrame(Some(t_nan))) =
                (out_base, out_nan)
            {
                let orig_cols = t_orig.size()[1];
                let t_nan_orig_part = t_nan.narrow(1, 0, orig_cols);

                // We use a small epsilon for floating point comparison.
                // We also need to handle NaN equivalence (equal_nan=true) in isclose.
                let is_close = t_orig.isclose(&t_nan_orig_part, 1e-4, 1e-6, true);
                let is_all_true = i64::try_from(is_close.all()).unwrap() != 0;
                assert!(
                    is_all_true,
                    "Operator {} failed nan instrument invariance test (valid columns changed)",
                    op_name
                );

                // let t_nan_added_part = t_nan.narrow(1, orig_cols, num_nan_cols);
                // let is_nan = t_nan_added_part.isnan();
                // let is_all_nan = i64::try_from(is_nan.all()).unwrap() != 0;
                // assert!(
                //     is_all_nan,
                //     "Operator {} failed nan instrument invariance test (added columns are not NaN)",
                //     op_name
                // );
            }
        }
    }
}
