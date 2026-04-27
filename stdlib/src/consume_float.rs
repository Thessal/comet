// src/stdlib/consume_float.rs

inventory::submit! {
    crate::OperatorMeta {
        name: "consume_float",
        inputs: &[crate::Signal::Float(None)],
        output_shape: crate::Signal::Void,
        execute: |_args: &[crate::Signal]| -> crate::Signal {
            crate::Signal::Void
        }
    }
}
