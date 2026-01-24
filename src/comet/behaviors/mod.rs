use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;

pub trait BehaviorHandler {
    fn handle(&self, synthesizer: &Synthesizer, args: &[Expr], context: &Context) -> Result<(String, Vec<String>), SynthesisError>;
}

pub mod normalizer;
