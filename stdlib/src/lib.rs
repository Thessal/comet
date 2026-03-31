// src/stdlib/lib.rs
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]

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
                inputs: &[crate::OutputShape::ScalarString],
                output_shape: crate::OutputShape::DataFrame,
                execute: |_args: &[crate::ParamType]| -> crate::ParamType {
                    crate::ParamType::Vector(vec![])
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputShape {
    Void,
    TimeSeries, // 1 element per timestep
    Vector,     // Variable elements (Iliffe vector)
    DataFrame,  // len elements per timestep
    Matrix,     // len * len elements per timestep
    ScalarFloat,
    ScalarInt,
    ScalarString,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamType {
    // Used to evaluate parameters in runtime
    Float(f64),
    String(String),
    Variable(String),
    Vector(Vec<f64>),
    DataFrame(Vec<Vec<f64>>),
}

#[derive(Clone)]
pub struct OperatorMeta {
    pub name: &'static str,
    pub inputs: &'static [OutputShape],
    pub output_shape: OutputShape,
    pub execute: fn(&[crate::ParamType]) -> crate::ParamType,
}

inventory::collect!(OperatorMeta);

#[macro_export]
macro_rules! register_unary_op {
    ($state:ident, $name:expr) => {
        inventory::submit! {
            crate::OperatorMeta {
                name: $name,
                inputs: &[crate::OutputShape::DataFrame],
                output_shape: crate::OutputShape::DataFrame,
                execute: |args: &[crate::ParamType]| -> crate::ParamType {
                    match &args[0] {
                        crate::ParamType::DataFrame(df) => {
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
                            crate::ParamType::DataFrame(outputs)
                        }
                        crate::ParamType::Vector(v) => {
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
                            crate::ParamType::Vector(outputs)
                        }
                        crate::ParamType::Float(f) => {
                            let mut state = $state::new(0, 1);
                            let mut output = vec![0.0; 1];
                            let a_data = crate::CometData {
                                dtype: crate::DataType::Constant,
                                ptr: f as *const f64,
                            };
                            state.step(a_data, output.as_mut_ptr());
                            crate::ParamType::Float(output[0])
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
                inputs: &[crate::OutputShape::DataFrame, crate::OutputShape::DataFrame],
                output_shape: crate::OutputShape::DataFrame,
                execute: |args: &[crate::ParamType]| -> crate::ParamType {
                    let t_len = match &args[0] {
                        crate::ParamType::DataFrame(df) => df.len(),
                        crate::ParamType::Vector(v) => v.len(),
                        _ => match &args[1] {
                            crate::ParamType::DataFrame(df) => df.len(),
                            crate::ParamType::Vector(v) => v.len(),
                            _ => 1,
                        }
                    };
                    let n_len = match &args[0] {
                        crate::ParamType::DataFrame(df) => if t_len > 0 { df[0].len() } else { 0 },
                        _ => match &args[1] {
                            crate::ParamType::DataFrame(df) => if t_len > 0 { df[0].len() } else { 0 },
                            _ => 1,
                        }
                    };

                    let is_output_df = matches!(&args[0], crate::ParamType::DataFrame(_)) || matches!(&args[1], crate::ParamType::DataFrame(_));

                    let mut state = $state::new(0, n_len);
                    let mut df_out = Vec::with_capacity(t_len);
                    let mut vec_out = Vec::with_capacity(t_len);

                    for t in 0..t_len {
                        let (a_ptr, a_const) = match &args[0] {
                            crate::ParamType::DataFrame(df) => (df[t].as_ptr(), false),
                            crate::ParamType::Vector(v) => (&v[t] as *const f64, true),
                            crate::ParamType::Float(f) => (f as *const f64, true),
                            _ => panic!("Expected Vector/DataFrame/Float"),
                        };
                        let (b_ptr, b_const) = match &args[1] {
                            crate::ParamType::DataFrame(df) => (df[t].as_ptr(), false),
                            crate::ParamType::Vector(v) => (&v[t] as *const f64, true),
                            crate::ParamType::Float(f) => (f as *const f64, true),
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
                        crate::ParamType::DataFrame(df_out)
                    } else if t_len == 1 && matches!(&args[0], crate::ParamType::Float(_)) && matches!(&args[1], crate::ParamType::Float(_)) {
                        crate::ParamType::Float(vec_out[0])
                    } else {
                        crate::ParamType::Vector(vec_out)
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
                inputs: &[crate::OutputShape::ScalarInt, crate::OutputShape::DataFrame],
                output_shape: crate::OutputShape::DataFrame,
                execute: |args: &[crate::ParamType]| -> crate::ParamType {
                    let t_len = match &args[1] {
                        crate::ParamType::DataFrame(df) => df.len(),
                        crate::ParamType::Vector(v) => v.len(),
                        crate::ParamType::Float(_) => 1,
                        _ => panic!("Expected Vector/DataFrame/Float"),
                    };
                    let n_len = match &args[1] {
                        crate::ParamType::DataFrame(df) => if t_len > 0 { df[0].len() } else { 0 },
                        _ => 1,
                    };

                    let is_output_df = matches!(&args[1], crate::ParamType::DataFrame(_));

                    let period = match &args[0] {
                        crate::ParamType::DataFrame(df) => {
                            if !df.is_empty() && !df[0].is_empty() { df[0][0] as usize } else { 0 }
                        }
                        crate::ParamType::Vector(v) => if !v.is_empty() { v[0] as usize } else { 0 },
                        crate::ParamType::Float(f) => *f as usize,
                        _ => panic!("Expected period as Vector/DataFrame/Float"),
                    };

                    let mut state = $state::new(period, n_len);
                    let mut df_out = Vec::with_capacity(t_len);
                    let mut vec_out = Vec::with_capacity(t_len);

                    for t in 0..t_len {
                        let (b_ptr, b_const) = match &args[1] {
                            crate::ParamType::DataFrame(df) => (df[t].as_ptr(), false),
                            crate::ParamType::Vector(v) => (&v[t] as *const f64, true),
                            crate::ParamType::Float(f) => (f as *const f64, true),
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
                        crate::ParamType::DataFrame(df_out)
                    } else if t_len == 1 && matches!(&args[1], crate::ParamType::Float(_)) {
                        crate::ParamType::Float(vec_out[0])
                    } else {
                        crate::ParamType::Vector(vec_out)
                    }
                }
            }
        }
    };
}
