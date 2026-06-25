use crate::{OperatorSpec, types::Signal};

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
                let res = a.roll(&[d_safe], &[0]);
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

