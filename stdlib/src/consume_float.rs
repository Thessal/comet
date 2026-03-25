// src/stdlib/consume_float.rs

inventory::submit! {
    crate::OperatorMeta {
        name: "consume_float",
        inputs: &[crate::OutputShape::ScalarFloat],
        output_shape: crate::OutputShape::Void,
        execute: |_args: &[std::vec::Vec<f64>]| -> std::vec::Vec<f64> {
            // Void function: does nothing and returns an empty vector
            vec![]
        }
    }
}
