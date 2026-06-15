use crate::{OperatorSpec, types::Signal};

pub static OP_CS_RANK: OperatorSpec = OperatorSpec {
    name: "cs_rank",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        match &args[0] {
            Signal::DataFrame(Some(a)) => {
                // Cross-sectional rank along dim 1
                let rank_fwd = a.argsort(1, false).argsort(1, false).to_kind(a.kind());
                let rank_rev = a
                    .flip([1])
                    .argsort(1, false)
                    .argsort(1, false)
                    .to_kind(a.kind())
                    .flip([1]);
                let rank = (rank_fwd + rank_rev) / 2.0; // Average for tied rank
                let rank = rank / ((a.size()[1] - 1) as f64);
                Signal::DataFrame(Some(rank))
            }
            _ => panic!("cs_rank expected DataFrame"),
        }
    },
};
