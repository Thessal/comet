use crate::comet::synthesis::{Synthesizer, Context, SynthesisError};
use crate::comet::ast::Expr;
use crate::comet::functions::FunctionHandler;
use crate::comet::ir::{ExecutionNode, OperatorOp};

pub struct Divide;

impl FunctionHandler for Divide {
    fn handle(&self, synthesizer: &Synthesizer, args: &Vec<Expr>, context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError> {
        if args.len() != 2 {
            return Err(SynthesisError::ConstraintFailed("divide requires exactly 2 arguments".to_string()));
        }

        let left_results = synthesizer.evaluate_expr(&args[0], context)?;
        
        let mut final_results = Vec::new();
        
        for (ctx, lhs_type, _, lhs_id) in left_results {
            let right_results = synthesizer.evaluate_expr(&args[1], ctx)?;
            
            for (mut ctx2, rhs_type, _, rhs_id) in right_results {
                let is_constant = |ty_name: &str| -> bool {
                     if let Some(info) = synthesizer.symbol_table.types.get(ty_name) {
                         return info.properties.iter().any(|p| p == "Constant");
                     }
                     false
                 };

                let lhs_const = is_constant(&lhs_type);
                let rhs_const = is_constant(&rhs_type);
                
                let mut res_type = "Unknown".to_string();

                if lhs_type == "DataFrame" {
                    if rhs_type == "DataFrame" || rhs_type == "TimeSeries" || rhs_const {
                       res_type = "DataFrame".to_string();
                    }
                } else if lhs_type == "TimeSeries" {
                    if rhs_type == "TimeSeries" || rhs_const {
                        res_type = "TimeSeries".to_string();
                    }
                } else if lhs_const {
                    if rhs_const {
                         res_type = lhs_type.clone(); 
                    }
                }
                
                if res_type == "Unknown" {
                    // Skip or Error? If branching, maybe one path is valid?
                    // For now, hard error if types don't match in this branch.
                    return Err(SynthesisError::TypeMismatch(format!("Compatible Division Types (LHS: {})", lhs_type), rhs_type));
                }
                
                let op_node = ExecutionNode::Operation {
                    op: OperatorOp::Divide,
                    args: vec![lhs_id, rhs_id],
                };
                let id = ctx2.add_node(op_node);
                final_results.push((ctx2, res_type, vec![], id));
            }
        }
        
        Ok(final_results)
    }
}
