// Minimal version of stdlib
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]
mod op_add;
mod op_cs_rank;
mod op_cs_zscore;
mod op_data;
mod op_divide;
mod op_flip;
mod op_multiply;
mod op_pow;
mod op_subtract;
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
            "cs_rank" => &op_cs_rank::OP_CS_RANK,
            "cs_zscore" => &op_cs_zscore::OP_CS_ZSCORE,
            "ts_mean" => &op_ts_mean::OP_TS_MEAN,
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
