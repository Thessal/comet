use crate::{OperatorSpec, types::Signal};

pub static OP_COS: OperatorSpec = OperatorSpec {
    name: "cos",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.cos())),
        _ => panic!("cos expected DataFrame"),
    },
};
