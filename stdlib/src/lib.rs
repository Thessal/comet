// src/stdlib/lib.rs

pub mod add;
pub mod cs_rank;
pub mod data;
pub mod subtract;
pub mod ts_diff;
pub mod consume_float;

#[cfg(test)]
mod test_cs_rank;
#[cfg(test)]
mod test_subtract;
#[cfg(test)]
mod test_ts_diff;

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
                execute: |_args: &[Vec<f64>]| -> Vec<f64> {
                    vec![]
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

#[derive(Clone)]
pub struct OperatorMeta {
    pub name: &'static str,
    pub inputs: &'static [OutputShape],
    pub output_shape: OutputShape,
    pub execute: fn(&[Vec<f64>]) -> Vec<f64>,
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
                execute: |args: &[Vec<f64>]| -> Vec<f64> {
                    let len = args[0].len();
                    let mut state = $state::new(0, len);
                    let mut output = vec![0.0; len];
                    let a_data = crate::CometData {
                        dtype: crate::DataType::DataFrame,
                        ptr: args[0].as_ptr(),
                    };
                    state.step(a_data, output.as_mut_ptr());
                    output
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
                execute: |args: &[Vec<f64>]| -> Vec<f64> {
                    let len = args[0].len();
                    let mut state = $state::new(0, len);
                    let mut output = vec![0.0; len];
                    let a_data = crate::CometData {
                        dtype: crate::DataType::DataFrame,
                        ptr: args[0].as_ptr(),
                    };
                    let b_data = crate::CometData {
                        dtype: crate::DataType::DataFrame,
                        ptr: args[1].as_ptr(),
                    };
                    state.step(a_data, b_data, output.as_mut_ptr());
                    output
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
                execute: |args: &[Vec<f64>]| -> Vec<f64> {
                    let len = args[1].len();
                    let period = if args[0].len() > 0 { args[0][0] as usize } else { 0 };
                    let mut state = $state::new(period, len);
                    let mut output = vec![0.0; len];
                    let a_data = crate::CometData {
                        dtype: crate::DataType::DataFrame,
                        ptr: args[1].as_ptr(),
                    };
                    state.step(a_data, output.as_mut_ptr());
                    output
                }
            }
        }
    };
}
