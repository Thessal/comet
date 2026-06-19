use crate::{OperatorSpec, types::Signal};
use tch::Tensor;

fn roll_window(a: &tch::Tensor, d: i64) -> tch::Tensor {
    let d_safe = std::cmp::max(1, d);
    let pad_len = d_safe - 1;
    if pad_len == 0 {
        return a.unfold(0, d_safe, 1);
    }
    let mut pad_shape = a.size();
    pad_shape[0] = pad_len;
    let nan_pad = tch::Tensor::full(&pad_shape, f64::NAN, (a.kind(), a.device()));
    let padded = tch::Tensor::cat(&[&nan_pad, a], 0);
    padded.unfold(0, d_safe, 1)
}

pub static OP_DELAY: OperatorSpec = OperatorSpec {
    name: "delay",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            let t_len = a.size()[0];

            if d < 0 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                Signal::DataFrame(Some(nan))
            } else if d == 0 {
                Signal::DataFrame(Some(a.shallow_clone()))
            } else {
                let d_safe = std::cmp::min(d, t_len);
                let mut res = a.roll(&[d_safe], &[0]);
                if d_safe > 0 && d_safe <= t_len {
                    let mut slice = res.narrow(0, 0, d_safe);
                    let nan = tch::Tensor::full(&[1], f64::NAN, (a.kind(), a.device()));
                    let _ = slice.copy_(&nan);
                }
                Signal::DataFrame(Some(res))
            }
        }
        _ => panic!("delay expected DataFrame and Int"),
    },
};

pub static OP_DELTA: OperatorSpec = OperatorSpec {
    name: "delta",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let a = &args[0];
        let delay_res = OP_DELAY.execute(args).unwrap();
        if let (Signal::DataFrame(Some(a_tensor)), Signal::DataFrame(Some(delay_tensor))) =
            (a, delay_res)
        {
            Signal::DataFrame(Some(a_tensor - delay_tensor))
        } else {
            panic!("delta execution failed");
        }
    },
};

pub static OP_TS_RETURN: OperatorSpec = OperatorSpec {
    name: "ts_return",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| {
        let a = &args[0];
        let delay_res = OP_DELAY.execute(args).unwrap();
        if let (Signal::DataFrame(Some(a_tensor)), Signal::DataFrame(Some(delay_tensor))) =
            (a, delay_res)
        {
            Signal::DataFrame(Some((a_tensor - &delay_tensor) / delay_tensor))
        } else {
            panic!("ts_return execution failed");
        }
    },
};

pub static OP_TS_MAX: OperatorSpec = OperatorSpec {
    name: "ts_max",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }
            let unfolded = roll_window(a, d);
            let res = unfolded
                .nan_to_num(
                    std::f64::NEG_INFINITY,
                    std::f64::NEG_INFINITY,
                    std::f64::NEG_INFINITY,
                )
                .amax(&[-1], false);
            let is_neginf = res.eq(std::f64::NEG_INFINITY);
            let nan = tch::Tensor::full(res.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(res.where_self(&is_neginf.logical_not(), &nan)))
        }
        _ => panic!("ts_max expected DataFrame and Int"),
    },
};

pub static OP_TS_MIN: OperatorSpec = OperatorSpec {
    name: "ts_min",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }
            let unfolded = roll_window(a, d);
            let res = unfolded
                .nan_to_num(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY)
                .amin(&[-1], false);
            let is_posinf = res.eq(std::f64::INFINITY);
            let nan = tch::Tensor::full(res.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(res.where_self(&is_posinf.logical_not(), &nan)))
        }
        _ => panic!("ts_min expected DataFrame and Int"),
    },
};

pub static OP_TS_ARGMAX: OperatorSpec = OperatorSpec {
    name: "ts_argmax",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 0 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }
            let window_size = d + 1;
            let unfolded = roll_window(a, window_size);
            let reversed = unfolded.flip([-1_i64]);
            let filled = reversed.nan_to_num(
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
                std::f64::NEG_INFINITY,
            );
            let argmax = filled.argmax(-1, false).to_kind(a.kind());
            let all_nan = unfolded.isnan().all_dim(-1, false);
            let nan = tch::Tensor::full(argmax.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(argmax.where_self(&all_nan.logical_not(), &nan)))
        }
        _ => panic!("ts_argmax expected DataFrame and Int"),
    },
};

pub static OP_TS_ARGMIN: OperatorSpec = OperatorSpec {
    name: "ts_argmin",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 0 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }
            let window_size = d + 1;
            let unfolded = roll_window(a, window_size);
            let reversed = unfolded.flip([-1_i64]);
            let filled =
                reversed.nan_to_num(std::f64::INFINITY, std::f64::INFINITY, std::f64::INFINITY);
            let argmin = filled.argmin(-1, false).to_kind(a.kind());
            let all_nan = unfolded.isnan().all_dim(-1, false);
            let nan = tch::Tensor::full(argmin.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(argmin.where_self(&all_nan.logical_not(), &nan)))
        }
        _ => panic!("ts_argmin expected DataFrame and Int"),
    },
};

pub static OP_TS_SUM: OperatorSpec = OperatorSpec {
    name: "ts_sum",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }
            let unfolded = roll_window(a, d);
            let sum = unfolded.sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            Signal::DataFrame(Some(sum))
        }
        _ => panic!("ts_sum expected DataFrame and Int"),
    },
};

pub static OP_TS_PROD: OperatorSpec = OperatorSpec {
    name: "ts_prod",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }
            let unfolded = roll_window(a, d);
            let prod = unfolded.prod_dim_int(-1, false, a.kind());
            Signal::DataFrame(Some(prod))
        }
        _ => panic!("ts_prod expected DataFrame and Int"),
    },
};

pub static OP_TS_NANMEAN: OperatorSpec = OperatorSpec {
    name: "ts_nanmean",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(a.shallow_clone()));
            }
            let unfolded = roll_window(a, d);
            let nansum = unfolded.nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(
                Some(&[-1][..]),
                false,
                a.kind(),
            );
            let count_valid =
                unfolded
                    .isnan()
                    .logical_not()
                    .sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let count_valid = count_valid.to_kind(a.kind());
            let mean = nansum / &count_valid;
            // where count_valid == 0, replace with NaN
            let nan = tch::Tensor::full(mean.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            let is_zero = count_valid.eq(0.0);
            Signal::DataFrame(Some(mean.where_self(&is_zero.logical_not(), &nan)))
        }
        _ => panic!("ts_nanmean expected DataFrame and Int"),
    },
};

pub static OP_TS_STDDEV: OperatorSpec = OperatorSpec {
    name: "ts_stddev",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 2 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            }
            let unfolded = roll_window(a, d);
            let nansum = unfolded.nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(
                Some(&[-1][..]),
                false,
                a.kind(),
            );
            let count_valid = unfolded
                .isnan()
                .logical_not()
                .to_kind(a.kind())
                .sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let mean = &nansum / &count_valid;
            let diff = &unfolded - mean.unsqueeze(-1);
            let var = diff.square().nan_to_num(0.0, 0.0, 0.0).sum_dim_intlist(
                Some(&[-1][..]),
                false,
                a.kind(),
            ) / (&count_valid - 1.0);
            let stddev = var.sqrt();
            let is_invalid = count_valid.less_equal(1.0);
            let nan = tch::Tensor::full(stddev.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(stddev.where_self(&is_invalid.logical_not(), &nan)))
        }
        _ => panic!("ts_stddev expected DataFrame and Int"),
    },
};

pub static OP_TS_RANK: OperatorSpec = OperatorSpec {
    name: "ts_rank",
    inputs: &[Signal::DataFrame(None), Signal::Int(None)],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1]) {
        (Signal::DataFrame(Some(a)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 1 {
                let nan = tch::Tensor::full(a.size().as_slice(), f64::NAN, (a.kind(), a.device()));
                return Signal::DataFrame(Some(nan));
            } else if d == 1 {
                return Signal::DataFrame(Some(tch::Tensor::full(
                    a.size().as_slice(),
                    1.0,
                    (a.kind(), a.device()),
                )));
            }
            let unfolded = roll_window(a, d);
            let last_elem = unfolded.narrow(-1, d - 1, 1);
            // Count how many valid elements are less than or equal to last_elem
            let less_equal = unfolded.less_equal_tensor(&last_elem);
            let valid = unfolded.isnan().logical_not();
            let valid_less_equal = less_equal.logical_and(&valid).to_kind(a.kind());
            let rank = valid_less_equal.sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let count_valid =
                valid
                    .to_kind(a.kind())
                    .sum_dim_intlist(Some(&[-1][..]), false, a.kind());
            let res = &rank / &count_valid;

            let is_last_nan = last_elem.isnan().squeeze_dim(-1);
            let nan = tch::Tensor::full(res.size().as_slice(), f64::NAN, (a.kind(), a.device()));
            Signal::DataFrame(Some(res.where_self(&is_last_nan.logical_not(), &nan)))
        }
        _ => panic!("ts_rank expected DataFrame and Int"),
    },
};

pub static OP_TS_COV: OperatorSpec = OperatorSpec {
    name: "ts_cov",
    inputs: &[
        Signal::DataFrame(None),
        Signal::DataFrame(None),
        Signal::Int(None),
    ],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1], &args[2]) {
        (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 2 {
                let nan = tch::Tensor::full(x.size().as_slice(), f64::NAN, (x.kind(), x.device()));
                return Signal::DataFrame(Some(nan));
            }
            let unfolded_x = roll_window(x, d);
            let unfolded_y = roll_window(y, d);

            let valid_x = unfolded_x.isnan().logical_not();
            let valid_y = unfolded_y.isnan().logical_not();
            let valid_both = valid_x.logical_and(&valid_y);
            let count_both =
                valid_both
                    .to_kind(x.kind())
                    .sum_dim_intlist(Some(&[-1][..]), false, x.kind());

            let zeros = tch::Tensor::zeros(&[1], (x.kind(), x.device()));
            let x_masked = unfolded_x.where_self(&valid_both, &zeros);
            let y_masked = unfolded_y.where_self(&valid_both, &zeros);

            let nansum_x = x_masked.sum_dim_intlist(Some(&[-1][..]), false, x.kind());
            let nansum_y = y_masked.sum_dim_intlist(Some(&[-1][..]), false, x.kind());

            let mean_x = &nansum_x / &count_both;
            let mean_y = &nansum_y / &count_both;

            let diff_x = &x_masked - mean_x.unsqueeze(-1);
            let diff_y = &y_masked - mean_y.unsqueeze(-1);

            let prod = (diff_x * diff_y).where_self(&valid_both, &zeros);
            let cov = prod.sum_dim_intlist(Some(&[-1][..]), false, x.kind()) / (&count_both - 1.0);

            let is_invalid = count_both.less_equal(1.0);
            let nan = tch::Tensor::full(cov.size().as_slice(), f64::NAN, (x.kind(), x.device()));
            Signal::DataFrame(Some(cov.where_self(&is_invalid.logical_not(), &nan)))
        }
        _ => panic!("ts_cov expected DataFrame, DataFrame and Int"),
    },
};

pub static OP_TS_CORR: OperatorSpec = OperatorSpec {
    name: "ts_corr",
    inputs: &[
        Signal::DataFrame(None),
        Signal::DataFrame(None),
        Signal::Int(None),
    ],
    output_shape: Signal::DataFrame(None),
    execute: |args| match (&args[0], &args[1], &args[2]) {
        (Signal::DataFrame(Some(x)), Signal::DataFrame(Some(y)), Signal::Int(Some(d))) => {
            let d = *d as i64;
            if d < 2 {
                let nan = tch::Tensor::full(x.size().as_slice(), f64::NAN, (x.kind(), x.device()));
                return Signal::DataFrame(Some(nan));
            }
            let unfolded_x = roll_window(x, d);
            let unfolded_y = roll_window(y, d);

            let valid_x = unfolded_x.isnan().logical_not();
            let valid_y = unfolded_y.isnan().logical_not();
            let valid_both = valid_x.logical_and(&valid_y);
            let count_both =
                valid_both
                    .to_kind(x.kind())
                    .sum_dim_intlist(Some(&[-1][..]), false, x.kind());

            let zeros = tch::Tensor::zeros(&[1], (x.kind(), x.device()));
            let x_masked = unfolded_x.where_self(&valid_both, &zeros);
            let y_masked = unfolded_y.where_self(&valid_both, &zeros);

            let mean_x = x_masked.sum_dim_intlist(Some(&[-1][..]), false, x.kind()) / &count_both;
            let mean_y = y_masked.sum_dim_intlist(Some(&[-1][..]), false, x.kind()) / &count_both;

            let diff_x = (&x_masked - mean_x.unsqueeze(-1)).where_self(&valid_both, &zeros);
            let diff_y = (&y_masked - mean_y.unsqueeze(-1)).where_self(&valid_both, &zeros);

            let cov = (&diff_x * &diff_y).sum_dim_intlist(Some(&[-1][..]), false, x.kind());
            let var_x = diff_x
                .square()
                .sum_dim_intlist(Some(&[-1][..]), false, x.kind());
            let var_y = diff_y
                .square()
                .sum_dim_intlist(Some(&[-1][..]), false, x.kind());

            let corr = cov / (var_x * var_y).sqrt();

            let is_invalid = count_both.less_equal(1.0);
            let nan = tch::Tensor::full(corr.size().as_slice(), f64::NAN, (x.kind(), x.device()));
            Signal::DataFrame(Some(corr.where_self(&is_invalid.logical_not(), &nan)))
        }
        _ => panic!("ts_corr expected DataFrame, DataFrame and Int"),
    },
};
