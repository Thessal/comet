use crate::{OperatorSpec, types::Signal};

pub static OP_DIVIDE: OperatorSpec = OperatorSpec {
    name: "divide",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::DataFrame(Some(b))) => {
            let result = a / b;
            let result = result.nan_to_num(Some(0.0), Some(1e9), Some(-1e9));
            Signal::DataFrame(Some(result))
        }
        _ => panic!("divide expected two DataFrames"),
    },
};
