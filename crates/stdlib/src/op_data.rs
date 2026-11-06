use crate::{OperatorSpec, types::Signal};

pub static OP_DATA: OperatorSpec = OperatorSpec {
    name: "data",
    inputs: &[Signal::String(None)],
    output_shape: Signal::DataFrame(None),
    execute: |_args| panic!("Data operator cannot be executed directly"),
};
