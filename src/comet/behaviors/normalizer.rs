use super::BehaviorHandler;
use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;

pub struct Normalizer;

impl BehaviorHandler for Normalizer {
    fn handle(&self, synthesizer: &Synthesizer, args: &[Expr], context: &Context) -> Result<(String, Vec<String>), SynthesisError> {
        if args.len() < 1 {
            return Err(SynthesisError::ConstraintFailed("Normalizer requires 1 argument".to_string()));
        }
        let (ty, mut props) = synthesizer.evaluate_expr(&args[0], context)?;
        props.push("Ranged".to_string());
        Ok((ty, props))
    }
}
