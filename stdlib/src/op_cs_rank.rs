use crate::{OperatorSpec, types::Signal};

pub static OP_CS_RANK: OperatorSpec = OperatorSpec {
    name: "cs_rank",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        match &args[0] {
            Signal::DataFrame(Some(a)) => {
                // Cross-sectional rank along dim 1
                let rank = a.argsort(1, false).argsort(1, false).to_kind(a.kind());
                Signal::DataFrame(Some(rank))
            }
            _ => panic!("cs_rank expected DataFrame"),
        }
    },
};
