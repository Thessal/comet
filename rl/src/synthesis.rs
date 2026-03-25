use parser::ast::{Ident, Expr, Literal, BehaviorDecl, TypeDecl};
use std::collections::{HashSet, HashMap};

pub fn substitute_expr(expr: &Expr, env: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Identifier(id) => {
            if let Some(val) = env.get(id) {
                // Recursively substitute
                substitute_expr(val, env)
            } else {
                expr.clone()
            }
        },
        Expr::List(exprs) => {
            Expr::List(exprs.iter().map(|e| substitute_expr(e, env)).collect())
        },
        Expr::Range { start, step, end } => {
            Expr::Range {
                start: Box::new(substitute_expr(start, env)),
                step: step.as_ref().map(|s| Box::new(substitute_expr(s, env))),
                end: Box::new(substitute_expr(end, env)),
            }
        },
        Expr::Call { path, args } => {
            let new_args = args.iter().map(|arg| {
                substitute_expr(arg, env)
            }).collect();
            Expr::Call {
                path: path.clone(),
                args: new_args,
            }
        },
        Expr::MemberAccess { target, field } => {
            Expr::MemberAccess {
                target: Box::new(substitute_expr(target, env)),
                field: field.clone(),
            }
        },
        _ => expr.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnSignature {
    pub name: Ident,
    pub args: Vec<(Ident, TypeDecl)>,
    pub return_type: TypeDecl,
}

use codegen::ir::RealExpr;

#[derive(Debug, Clone)]
pub struct SubtreeState {
    pub tree: RealExpr,
    pub consumed_args: HashSet<Ident>,
    pub output_type: TypeDecl,
    pub depth: u32,
}

pub struct Synthesizer;

impl Synthesizer {
    pub fn synthesize_expr(expr: &Expr, behaviors: &HashMap<String, BehaviorDecl>, library: &[FnSignature]) -> Result<Vec<Vec<RealExpr>>, String> {
        match expr {
            Expr::Literal(lit) => Ok(vec![vec![RealExpr::Literal(lit.clone())]]),
            Expr::Identifier(id) => Ok(vec![vec![RealExpr::Identifier(id.clone())]]),
            Expr::List(exprs) => {
                let mut possibilities = Vec::new();
                for e in exprs {
                    let mut e_combs = Self::synthesize_expr(e, behaviors, library)?;
                    possibilities.append(&mut e_combs);
                }
                Ok(possibilities)
            },
            Expr::Range { start, step, end } => {
                let start_lit = Self::get_lit(start)?;
                let end_lit = Self::get_lit(end)?;
                let step_lit = if let Some(st) = step {
                    Self::get_lit(st)?
                } else {
                    if let Literal::Integer(_) = start_lit { Literal::Integer(1) } else { Literal::Float(1.0) }
                };

                let mut possibilities = Vec::new();
                match (start_lit, end_lit, step_lit) {
                    (Literal::Integer(s), Literal::Integer(e), Literal::Integer(st)) => {
                        let mut current = s;
                        while current <= e {
                            possibilities.push(vec![RealExpr::Literal(Literal::Integer(current))]);
                            current += st;
                        }
                    },
                    (Literal::Float(s), Literal::Float(e), Literal::Float(st)) => {
                        let mut current = s;
                        let epsilon = 1e-9;
                        while current <= e + epsilon {
                            possibilities.push(vec![RealExpr::Literal(Literal::Float(current))]);
                            current += st;
                        }
                    },
                    _ => return Err("Range bounds must be monotonically matching numeric literals (Ints or Floats)".to_string()),
                }
                Ok(possibilities)
            },
            Expr::Call { path, args } => {
                // Cartesian product of arguments!
                let mut arg_possibilities: Vec<Vec<RealExpr>> = vec![vec![]];
                
                for arg in args {
                    let evaluated_values = Self::synthesize_expr(arg, behaviors, library)?;
                    
                    let mut next_product = Vec::new();
                    for partial_tuple in &arg_possibilities {
                        for val_forest in &evaluated_values {
                            let mut new_tuple = partial_tuple.clone();
                            new_tuple.push(val_forest[0].clone());
                            next_product.push(new_tuple);
                        }
                    }
                    arg_possibilities = next_product;
                }

                let mut call_combinations = Vec::new();
                let func_name = path.segments.last().unwrap().clone();
                for args_tuple in arg_possibilities {
                    call_combinations.push(vec![RealExpr::CallFn {
                        func_name: func_name.clone(),
                        args: args_tuple,
                        return_type: TypeDecl::Bool,
                    }]);
                }
                Ok(call_combinations)
            },
            _ => Err("Unsupported AST Expression for Synthesis Cartesian Generation".to_string())
        }
    }

    fn get_lit(expr: &Expr) -> Result<Literal, String> {
         match expr {
             Expr::Literal(l) => Ok(l.clone()),
             _ => Err("Not a literal".to_string())
         }
    }
}

