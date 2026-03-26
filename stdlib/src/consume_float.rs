// src/stdlib/consume_float.rs

inventory::submit! {
    crate::OperatorMeta {
        name: "consume_float",
        inputs: &[crate::OutputShape::ScalarFloat],
        output_shape: crate::OutputShape::Void,
        execute: |_args: &[crate::ParamType]| -> crate::ParamType {
            // Void function: does nothing and returns an empty vector
            crate::ParamType::Vector(vec![])
        }
    }
}
