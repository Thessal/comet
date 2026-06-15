use crate::{OperatorSpec, types::Signal};

pub static OP_POW: OperatorSpec = OperatorSpec {
    name: "pow",
    inputs: &[Signal::DataFrame(None), Signal::Float(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Float(Some(x))) => {
            Signal::DataFrame(Some(a.pow_tensor_scalar(*x)))
        }
        _ => panic!("pow expected DataFrame and Float"),
    },
};
