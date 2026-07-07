use crate::{OperatorSpec, types::Signal};

pub static OP_SQRT: OperatorSpec = OperatorSpec {
    name: "sqrt",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.sqrt())),
        _ => panic!("sqrt expected DataFrame"),
    },
};
