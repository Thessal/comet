use crate::{OperatorSpec, types::Signal};

pub static OP_CS_ZSCORE: OperatorSpec = OperatorSpec {
    name: "cs_zscore",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => {
            let nansum =
                a.nan_to_num(0.0, 0.0, 0.0)
                    .sum_dim_intlist(Some(&[1][..]), true, a.kind());
            let count_valid = a.isnan().logical_not().to_kind(a.kind()).sum_dim_intlist(
                Some(&[1][..]),
                true,
                a.kind(),
            );
            let mean = &nansum / &count_valid;
            let diff = a - &mean;
            let var = diff.square().nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(
                Some(&[1][..]),
                true,
                a.kind(),
            ) / &count_valid;
            let std = var.sqrt();
            Signal::DataFrame(Some(diff / (std + 1e-10)))
        }
        _ => panic!("cs_zscore expected DataFrame"),
    },
};
