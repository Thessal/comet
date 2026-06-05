use crate::{OperatorSpec, types::Signal};

pub static OP_FLIP: OperatorSpec = OperatorSpec {
    name: "flip",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a * -1.0)),
        _ => panic!("flip expected a DataFrame"),
    },
};
