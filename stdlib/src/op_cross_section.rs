use crate::{OperatorSpec, types::Signal};

pub static OP_RANK: OperatorSpec = OperatorSpec {
    name: "rank",
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
                let rank = (rank_fwd + rank_rev) / 2.0;

                // Rank 0 to N-1. Normalize it to 0-1
                let valid = a.isnan().logical_not();
                let count_valid =
                    valid
                        .to_kind(a.kind())
                        .sum_dim_intlist(Some(&[1][..]), false, a.kind());
                let rank = rank / (count_valid.unsqueeze(1) - 1.0).clamp_min(1.0);

                let nan =
                    tch::Tensor::full(rank.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                Signal::DataFrame(Some(rank.where_self(&valid, &nan)))
            }
            _ => panic!("rank expected DataFrame"),
        }
    },
};

pub static OP_RANK_ADD: OperatorSpec = OperatorSpec {
    name: "rank_add",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let rank_x = OP_RANK.execute(&[args[0].clone()]).unwrap();
        let rank_y = OP_RANK.execute(&[args[1].clone()]).unwrap();
        if let (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y))) = (rank_x, rank_y) {
            Signal::DataFrame(Some(x + y))
        } else {
            panic!("rank_add execution failed");
        }
    },
};

pub static OP_RANK_SUB: OperatorSpec = OperatorSpec {
    name: "rank_sub",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let rank_x = OP_RANK.execute(&[args[0].clone()]).unwrap();
        let rank_y = OP_RANK.execute(&[args[1].clone()]).unwrap();
        if let (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y))) = (rank_x, rank_y) {
            Signal::DataFrame(Some(x - y))
        } else {
            panic!("rank_sub execution failed");
        }
    },
};

pub static OP_RANK_MUL: OperatorSpec = OperatorSpec {
    name: "rank_mul",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let rank_x = OP_RANK.execute(&[args[0].clone()]).unwrap();
        let rank_y = OP_RANK.execute(&[args[1].clone()]).unwrap();
        if let (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y))) = (rank_x, rank_y) {
            Signal::DataFrame(Some(x * y))
        } else {
            panic!("rank_mul execution failed");
        }
    },
};

pub static OP_RANK_DIV: OperatorSpec = OperatorSpec {
    name: "rank_div",
    inputs: &[Signal::DataFrame(None), Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let rank_x = OP_RANK.execute(&[args[0].clone()]).unwrap();
        let rank_y = OP_RANK.execute(&[args[1].clone()]).unwrap();
        if let (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y))) = (rank_x, rank_y) {
            Signal::DataFrame(Some(x / y))
        } else {
            panic!("rank_div execution failed");
        }
    },
};

pub static OP_SIGN: OperatorSpec = OperatorSpec {
    name: "sign",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.sign())),
        _ => panic!("sign expected DataFrame"),
    },
};

pub static OP_SIGMOID: OperatorSpec = OperatorSpec {
    name: "sigmoid",
    inputs: &[Signal::DataFrame(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match &args[0] {
        Signal::DataFrame(Some(a)) => Signal::DataFrame(Some(a.sigmoid())),
        _ => panic!("sigmoid expected DataFrame"),
    },
};
