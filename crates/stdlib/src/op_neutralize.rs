use crate::{OperatorSpec, types::Signal};
use tch::Kind;

pub static OP_NEUTRALIZE: OperatorSpec = OperatorSpec {
    name: "neutralize",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => {
            let mean = a.nanmean(Some(&[1][..]), true, Kind::Float);
            Signal::DataFrame(Some(a - mean))
        }
        _ => panic!("neutralize expected a DataFrame"),
    },
};
// TODO: ls neutralize
