use crate::{OperatorSpec, types::Signal};

pub static OP_ADD: OperatorSpec = OperatorSpec {
    name: "add",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::DataFrame(Some(b))) => Signal::DataFrame(Some(a + b)),
        _ => panic!("add expected two DataFrames"),
    },
};
