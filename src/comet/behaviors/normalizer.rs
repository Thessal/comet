use super::BehaviorHandler;
use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;

use crate::comet::ir::{ExecutionNode, OperatorOp};

pub struct Normalizer;

impl BehaviorHandler for Normalizer {
    fn handle(&self, synthesizer: &Synthesizer, args: &Vec<Expr>, context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError> {
        if args.len() < 1 {
            return Err(SynthesisError::ConstraintFailed("Normalizer requires 1 argument".to_string()));
        }
        let results = synthesizer.evaluate_expr(&args[0], context)?;
        
        let mut final_results = Vec::new();
        for (mut ctx, ty, mut props, id) in results {
            let op_node = ExecutionNode::Operation {
                op: OperatorOp::ZScore,
                args: vec![id],
            };
            let new_id = ctx.add_node(op_node);
            
            props.push("Ranged".to_string());
            final_results.push((ctx, ty, props, new_id));
        }
        Ok(final_results)
    }
}
