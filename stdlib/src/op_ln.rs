use crate::{OperatorSpec, types::Signal};

pub static OP_LN: OperatorSpec = OperatorSpec {
    name: "ln",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.log())),
        _ => panic!("ln expected DataFrame"),
    },
};
