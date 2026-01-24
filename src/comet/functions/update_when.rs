use super::FunctionHandler;
use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;

pub struct UpdateWhen;

impl FunctionHandler for UpdateWhen {
    fn handle(&self, synthesizer: &Synthesizer, args: &[Expr], context: &Context) -> Result<(String, Vec<String>), SynthesisError> {
        // Signature: update_when(data, signal, cond) where signal is Ranged
        // Data is arg 0, Signal is arg 1
        
        if args.len() < 2 {
             // In real generic handling we'd check against signature, but here we hardcode checks
        }

        if let Some(signal_expr) = args.get(1) {
            let (_, signal_props) = synthesizer.evaluate_expr(signal_expr, context)?;
            if !signal_props.contains(&"Ranged".to_string()) {
                return Err(SynthesisError::ConstraintFailed("Signal in update_when must be Ranged".to_string()));
            }
        }
        // Return type is type of arg 0
        if let Some(arg0) = args.get(0) {
            synthesizer.evaluate_expr(arg0, context)
        } else {
             Err(SynthesisError::ConstraintFailed("update_when missing arguments".to_string()))
        }
    }
}
