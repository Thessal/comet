#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]
mod types;
use types::Signal;

pub mod add;
pub mod consume_float;
pub mod cs_rank;
pub mod cs_zscore;
pub mod data;
pub mod divide;
pub mod multiply;
pub mod subtract;
pub mod ts_diff;
pub mod ts_mean;
pub mod ts_rank;
pub mod ts_zscore;

use std::collections::HashMap;
use std::sync::OnceLock;

#[cfg(test)]
mod test_cs_rank;
#[cfg(test)]
mod test_cs_zscore;
#[cfg(test)]
mod test_subtract;
#[cfg(test)]
mod test_ts_diff;
#[cfg(test)]
mod test_ts_rank;
#[cfg(test)]
mod test_ts_zscore;

use std::collections::VecDeque;

pub trait PartialDeque {
    fn push(&mut self, new_data: &[f64]);
}

/// A generic-purpose ring buffer structured around VecDeque for safe Rust-native operations.
/// Rows = Time (capacity), Cols = Cross-section (len).
#[derive(Debug, Clone)]
pub struct DequeState {
    pub history: VecDeque<Vec<f64>>,
    pub col: usize,
    pub cap: usize,
    pub count: usize,
}

impl DequeState {
    pub fn new(capacity: usize, length: usize) -> Self {
        DequeState {
            history: VecDeque::with_capacity(if capacity == 0 { 1 } else { capacity + 1 }),
            col: length,
            cap: capacity,
            count: 0,
        }
    }

    /// Gets the oldest row (the one that will be overwritten next).
    pub fn get_oldest(&self) -> Option<&[f64]> {
        if self.count < self.cap || self.cap == 0 {
            return None;
        }
        self.history.front().map(|v| v.as_slice())
    }

    pub fn get_latest(&self) -> Option<&[f64]> {
        if self.count == 0 {
            return None;
        }
        self.history.back().map(|v| v.as_slice())
    }

    pub fn get_latest_mut(&mut self) -> Option<&mut [f64]> {
        if self.count == 0 {
            return None;
        }
        self.history.back_mut().map(|v| v.as_mut_slice())
    }
}

impl PartialDeque for DequeState {
    fn push(&mut self, new_data: &[f64]) {
        if self.cap == 0 || self.col == 0 {
            return;
        }
        assert_eq!(
            new_data.len(),
            self.col,
            "new_data len must match state col"
        );

        if self.count < self.cap {
            self.count += 1;
            self.history.push_back(new_data.to_vec());
        } else {
            // Re-use the oldest allocation for performance zero-copy where possible
            let mut oldest = self.history.pop_front().unwrap();
            oldest.copy_from_slice(new_data);
            self.history.push_back(oldest);
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Constant = 0,
    TimeSeries = 1,
    DataFrame = 2,
}
#[macro_export]
macro_rules! register_data_op {
    ($name:expr) => {
        inventory::submit! {
            crate::OperatorMeta {
                name: $name,
                inputs: &[crate::Signal::String(None)],
                output_shape: crate::Signal::DataFrame(None),
                execute: |_args: &[crate::Signal]| -> crate::Signal {
                    crate::Signal::DataFrame(Some(vec![]))
                }
            }
        }
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CometData {
    pub dtype: DataType,
    pub ptr: *const f64,
}

impl CometData {
    #[inline(always)]
    pub unsafe fn as_slice(&self, len: usize) -> &[f64] {
        if self.dtype == DataType::DataFrame {
            unsafe { std::slice::from_raw_parts(self.ptr, len) }
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, 1) }
        }
    }

    #[inline(always)]
    pub unsafe fn get_scalar(&self) -> f64 {
        unsafe { *self.ptr }
    }
}

// In src/stdlib/lib.rs
pub trait ZeroAryOp {
    fn step(&mut self, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

pub trait UnaryOp {
    fn step(&mut self, a: CometData, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

pub trait BinaryOp {
    fn step(&mut self, a: CometData, b: CometData, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

pub trait TernaryOp {
    fn step(&mut self, a: CometData, b: CometData, c: CometData, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

pub trait DataFrameOp {
    fn step(&mut self, a: CometData, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

pub trait MatrixOp {
    fn step(&mut self, a: CometData, out_ptr: *mut f64);
    fn drop_buffers(&mut self) {}
}

// #[derive(Debug, Clone, PartialEq)]
// pub enum Signal {
//     // Used to evaluate parameters in runtime
//     Void,
//     Float(Option<f64>),
//     Int(Option<i64>),
//     String(Option<String>),
//     Vector(Option<Vec<f64>>),
//     DataFrame(Option<Vec<Vec<f64>>>),
// }

#[derive(Clone)]
pub struct OperatorMeta {
    pub name: &'static str,
    pub inputs: &'static [Signal],
    pub output_shape: Signal,
    pub execute: fn(&[Signal]) -> Signal,
}

fn get_operator_meta_map() -> &'static HashMap<&'static str, OperatorMeta> {
    static MAP: OnceLock<HashMap<&'static str, OperatorMeta>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m = HashMap::new();
        for meta in inventory::iter::<OperatorMeta> {
            m.insert(meta.name, meta.clone());
        }
        m
    })
}

impl From<&str> for OperatorMeta {
    fn from(sig_name: &str) -> Self {
        if let Some(meta) = get_operator_meta_map().get(sig_name) {
            return meta.clone();
        }
        panic!("Could not find {} in the stdlib", sig_name);
    }
}

impl OperatorMeta {
    pub fn execute(&self, args: &[Signal]) -> Result<Signal, String> {
        let arity = self.inputs.len();
        if args.len() < arity {
            return Err(format!("Stack underflow for {}", self.name));
        }

        for (arg, expected) in args.iter().zip(self.inputs.iter()) {
            if std::mem::discriminant(arg) != std::mem::discriminant(expected) {
                return Err(format!(
                    "Type mistmatch for {}, arg: {:?}, expected: {:?}",
                    self.name, arg, expected
                ));
            }
        }

        Ok((self.execute)(args))
    }
}

inventory::collect!(OperatorMeta);

///////
// Calculating operator for dataframe
///////

#[macro_export]
macro_rules! register_unary_op {
    ($state:ident, $name:expr) => {
        inventory::submit! {
            crate::OperatorMeta {
                name: $name,
                inputs: &[crate::Signal::DataFrame(None)],
                output_shape: crate::Signal::DataFrame(None),
                execute: |args: &[crate::Signal]| -> crate::Signal {
                    match &args[0] {
                        crate::Signal::DataFrame(Some(df)) => {
                            let t_len = df.len();
                            let n_len = if t_len > 0 { df[0].len() } else { 0 };
                            let mut state = $state::new(0, n_len);
                            let mut outputs = Vec::with_capacity(t_len);
                            for t in 0..t_len {
                                let mut output = vec![0.0; n_len];
                                let a_data = crate::CometData {
                                    dtype: crate::DataType::DataFrame,
                                    ptr: df[t].as_ptr(),
                                };
                                state.step(a_data, output.as_mut_ptr());
                                outputs.push(output);
                            }
                            crate::Signal::DataFrame(Some(outputs))
                        }
                        crate::Signal::Vector(Some(v)) => {
                            let t_len = v.len();
                            let mut state = $state::new(0, 1);
                            let mut outputs = Vec::with_capacity(t_len);
                            for t in 0..t_len {
                                let mut output = vec![0.0; 1];
                                let a_data = crate::CometData {
                                    dtype: crate::DataType::DataFrame,
                                    ptr: &v[t] as *const f64,
                                };
                                state.step(a_data, output.as_mut_ptr());
                                outputs.push(output[0]);
                            }
                            crate::Signal::Vector(Some(outputs))
                        }
                        crate::Signal::Float(Some(f)) => {
                            let mut state = $state::new(0, 1);
                            let mut output = vec![0.0; 1];
                            let a_data = crate::CometData {
                                dtype: crate::DataType::Constant,
                                ptr: f as *const f64,
                            };
                            state.step(a_data, output.as_mut_ptr());
                            crate::Signal::Float(Some(output[0]))
                        }
                        _ => panic!("Expected Vector, DataFrame or Float"),
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! register_binary_op {
    ($state:ident, $name:expr) => {
        inventory::submit! {
            crate::OperatorMeta {
                name: $name,
                inputs: &[crate::Signal::DataFrame(None), crate::Signal::DataFrame(None)],
                output_shape: crate::Signal::DataFrame(None),
                execute: |args: &[crate::Signal]| -> crate::Signal {
                    let t_len = match &args[0] {
                        crate::Signal::DataFrame(Some(df)) => df.len(),
                        crate::Signal::Vector(Some(v)) => v.len(),
                        _ => match &args[1] {
                            crate::Signal::DataFrame(Some(df)) => df.len(),
                            crate::Signal::Vector(Some(v)) => v.len(),
                            _ => 1,
                        }
                    };
                    let n_len = match &args[0] {
                        crate::Signal::DataFrame(Some(df)) => if t_len > 0 { df[0].len() } else { 0 },
                        _ => match &args[1] {
                            crate::Signal::DataFrame(Some(df)) => if t_len > 0 { df[0].len() } else { 0 },
                            _ => 1,
                        }
                    };

                    let is_output_df = matches!(&args[0], crate::Signal::DataFrame(_)) || matches!(&args[1], crate::Signal::DataFrame(_));

                    let mut state = $state::new(0, n_len);
                    let mut df_out = Vec::with_capacity(t_len);
                    let mut vec_out = Vec::with_capacity(t_len);

                    for t in 0..t_len {
                        let (a_ptr, a_const) = match &args[0] {
                            crate::Signal::DataFrame(Some(df)) => (df[t].as_ptr(), false),
                            crate::Signal::Vector(Some(v)) => (&v[t] as *const f64, true),
                            crate::Signal::Float(Some(f)) => (f as *const f64, true),
                            _ => panic!("Expected Vector/DataFrame/Float"),
                        };
                        let (b_ptr, b_const) = match &args[1] {
                            crate::Signal::DataFrame(Some(df)) => (df[t].as_ptr(), false),
                            crate::Signal::Vector(Some(v)) => (&v[t] as *const f64, true),
                            crate::Signal::Float(Some(f)) => (f as *const f64, true),
                            _ => panic!("Expected Vector/DataFrame/Float"),
                        };

                        let mut output = vec![0.0; n_len];
                        let a_data = crate::CometData {
                            dtype: if a_const { crate::DataType::Constant } else { crate::DataType::DataFrame },
                            ptr: a_ptr,
                        };
                        let b_data = crate::CometData {
                            dtype: if b_const { crate::DataType::Constant } else { crate::DataType::DataFrame },
                            ptr: b_ptr,
                        };
                        state.step(a_data, b_data, output.as_mut_ptr());

                        if is_output_df {
                            df_out.push(output);
                        } else {
                            vec_out.push(output[0]);
                        }
                    }

                    if is_output_df {
                        crate::Signal::DataFrame(Some(df_out))
                    } else if t_len == 1 && matches!(&args[0], crate::Signal::Float(Some(_))) && matches!(&args[1], crate::Signal::Float(Some(_))) {
                        crate::Signal::Float(Some(vec_out[0]))
                    } else {
                        crate::Signal::Vector(Some(vec_out))
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! register_period_unary_op {
    ($state:ident, $name:expr) => {
        inventory::submit! {
            crate::OperatorMeta {
                name: $name,
                inputs: &[crate::Signal::Int(None), crate::Signal::DataFrame(None)],
                output_shape: crate::Signal::DataFrame(None),
                // TODO: FIXME
                execute: |args: &[crate::Signal]| -> crate::Signal {
                    let t_len = match &args[1] {
                        crate::Signal::DataFrame(Some(df)) => df.len(),
                        crate::Signal::Vector(Some(v)) => v.len(),
                        crate::Signal::Float(_) => 1,
                        _ => panic!("Expected Vector/DataFrame/Float"),
                    };
                    let n_len = match &args[1] {
                        crate::Signal::DataFrame(Some(df)) => if t_len > 0 { df[0].len() } else { 0 },
                        _ => 1,
                    };

                    let is_output_df = matches!(&args[1], crate::Signal::DataFrame(_));

                    let period = match &args[0] {
                        crate::Signal::DataFrame(Some(df)) => {
                            if !df.is_empty() && !df[0].is_empty() { df[0][0] as usize } else { 0 }
                        }
                        crate::Signal::Vector(Some(v)) => if !v.is_empty() { v[0] as usize } else { 0 },
                        crate::Signal::Float(Some(f)) => *f as usize,
                        crate::Signal::Int(Some(f)) => *f as usize,
                        _ => panic!("Expected period as Vector/DataFrame/Float"),
                    };

                    let mut state = $state::new(period, n_len);
                    let mut df_out = Vec::with_capacity(t_len);
                    let mut vec_out = Vec::with_capacity(t_len);

                    for t in 0..t_len {
                        let (b_ptr, b_const) = match &args[1] {
                            crate::Signal::DataFrame(Some(df)) => (df[t].as_ptr(), false),
                            crate::Signal::Vector(Some(v)) => (&v[t] as *const f64, true),
                            crate::Signal::Float(Some(f)) => (f as *const f64, true),
                            _ => panic!("Expected Vector/DataFrame/Float"),
                        };

                        let mut output = vec![0.0; n_len];
                        let a_data = crate::CometData {
                            dtype: if b_const { crate::DataType::Constant } else { crate::DataType::DataFrame },
                            ptr: b_ptr,
                        };
                        state.step(a_data, output.as_mut_ptr());

                        if is_output_df {
                            df_out.push(output);
                        } else {
                            vec_out.push(output[0]);
                        }
                    }

                    if is_output_df {
                        crate::Signal::DataFrame(Some(df_out))
                    } else if t_len == 1 && matches!(&args[1], crate::Signal::Float(Some(_))) {
                        crate::Signal::Float(Some(vec_out[0]))
                    } else {
                        crate::Signal::Vector(Some(vec_out))
                    }
                }
            }
        }
    };
}
