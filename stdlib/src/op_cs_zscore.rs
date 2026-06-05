use tch::Kind;

use crate::{OperatorSpec, types::Signal};

pub static OP_CS_ZSCORE: OperatorSpec = OperatorSpec {
    name: "cs_zscore",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => {
            let mean = a.mean_dim(1, true, Kind::Float);
            let std = a.std_dim(1, false, true);
            Signal::DataFrame(Some((a - mean) / (std + 1e-10)))
        }
        _ => panic!("cs_zscore expected DataFrame"),
    },
};
