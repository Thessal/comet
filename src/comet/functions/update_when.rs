use crate::comet::functions::FunctionHandler;
use crate::comet::synthesis::{Synthesizer, SynthesisError, Context};
use crate::comet::ast::Expr;
use crate::comet::ir::{ExecutionNode, OperatorOp};

pub struct UpdateWhen;

impl FunctionHandler for UpdateWhen {
    fn handle(&self, synthesizer: &Synthesizer, args: &Vec<Expr>, context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError> {
        // Signature: update_when(data, signal, cond) where signal is Ranged
        // Data is arg 0, Signal is arg 1
        
        if args.len() < 2 {
             // In real generic handling we'd check against signature, but here we hardcode checks
        }

        // We need to evaluate arg0, then arg1, then check properties.
        if let Some(arg0) = args.get(0) {
             let results0 = synthesizer.evaluate_expr(arg0, context)?;
             
             let mut final_results = Vec::new();

             for (ctx, t, props, id0) in results0 {
                 if let Some(arg1) = args.get(1) {
                      let results1 = synthesizer.evaluate_expr(arg1, ctx)?;
                      
                      for (mut ctx2, _, signal_props, id1) in results1 {
                          if !signal_props.contains(&"Ranged".to_string()) {
                              return Err(SynthesisError::ConstraintFailed("Signal in update_when must be Ranged".to_string()));
                          }
                          
                          let mut op_args = vec![id0];
                          op_args.push(id1);
                          
                          let op_node = ExecutionNode::Operation {
                             op: OperatorOp::UpdateWhen,
                             args: op_args,
                         };
                         let new_id = ctx2.add_node(op_node);
                         final_results.push((ctx2, t.clone(), props.clone(), new_id));
                      }
                 }
             }
             Ok(final_results)
        } else {
             Err(SynthesisError::ConstraintFailed("update_when missing arguments".to_string()))
        }
    }
}
