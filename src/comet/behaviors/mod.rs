use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;

pub trait BehaviorHandler {
    fn handle(&self, synthesizer: &Synthesizer, args: &Vec<Expr>, context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError>;
}

pub mod normalizer;
