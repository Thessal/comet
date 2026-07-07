use crate::{OperatorSpec, types::Signal};

pub static OP_SIN: OperatorSpec = OperatorSpec {
    name: "sin",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.sin())),
        _ => panic!("sin expected DataFrame"),
    },
};
