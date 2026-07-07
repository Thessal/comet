use crate::{OperatorSpec, types::Signal};

pub static OP_EXP: OperatorSpec = OperatorSpec {
    name: "exp",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.exp())),
        _ => panic!("exp expected DataFrame"),
    },
};
