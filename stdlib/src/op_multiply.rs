use crate::{OperatorSpec, types::Signal};

pub static OP_MULTIPLY: OperatorSpec = OperatorSpec {
    name: "multiply",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::DataFrame(Some(b))) => Signal::DataFrame(Some(a * b)),
        _ => panic!("multiply expected two DataFrames"),
    },
};
