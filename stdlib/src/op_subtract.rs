use crate::{OperatorSpec, types::Signal};
use tch::Tensor;

pub static OP_SUBTRACT: OperatorSpec = OperatorSpec {
    name: "subtract",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::DataFrame(Some(b))) => {
            // O(1) Pointer comparison: checks if references or underlying data buffers match (< 1 ns)
            if std::ptr::eq(a, b) || a.data_ptr() == b.data_ptr() {
                return Signal::DataFrame(Some(Tensor::zeros_like(a)));
            }
            Signal::DataFrame(Some(a - b))
        }
        _ => panic!("subtract expected two DataFrames"),
    },
};
