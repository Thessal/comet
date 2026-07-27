use crate::{OperatorSpec, types::Signal};
use tch::Tensor;

pub static OP_DIVIDE: OperatorSpec = OperatorSpec {
    name: "divide",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::DataFrame(Some(b))) => {
            // O(1) Pointer comparison: checks if references or underlying data buffers match (< 1 ns)
            if std::ptr::eq(a, b) || a.data_ptr() == b.data_ptr() {
                return Signal::DataFrame(Some(Tensor::ones_like(a)));
            }
            let result = a / b;
            let result = result.nan_to_num(Some(0.0), Some(1e9), Some(-1e9));
            Signal::DataFrame(Some(result))
        }
        _ => panic!("divide expected two DataFrames"),
    },
};
