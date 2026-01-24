use std::collections::HashMap;
use crate::comet::ast::{FlowDecl, FlowStmt, Expr, Ident, Program, Declaration};
use crate::comet::symbols::{SymbolTable, TypeInfo, BehaviorInfo, ImplInfo};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SynthesisError {
    #[error("Flow not found: {0}")]
    FlowNotFound(String),
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    #[error("Type mismatch: expected {0}, found {1}")]
    TypeMismatch(String, String),
    #[error("Constraint failed: {0}")]
    ConstraintFailed(String),
    #[error("Ambiguous implementation for behavior {0}")]
    AmbiguousImpl(String),
    #[error("No implementation found for behavior {0}")]
    NoImplFound(String),
}

use crate::comet::ir::{ExecutionGraph, ExecutionNode, OperatorOp};

#[derive(Debug, Clone)]
pub struct Context {
    // Map variable name to its "Type" or "Semantic State"
    pub variables: HashMap<Ident, VariableState>,
    pub graph: ExecutionGraph,
}

impl Context {
    pub fn new() -> Self {
        Context {
            variables: HashMap::new(),
            graph: ExecutionGraph::new(),
        }
    }
    
    pub fn add_node(&mut self, node: ExecutionNode) -> usize {
        self.graph.add_node(node)
    }
}

#[derive(Debug, Clone)]
pub struct VariableState {
    pub name: Ident,
    pub type_name: String, // e.g. "PERatio", "DataFrame"
    pub properties: Vec<String>, // e.g. ["Monetary", "Ranged"]
    pub node_id: usize,
}

pub struct Synthesizer<'a> {
    pub symbol_table: &'a SymbolTable,
}

impl<'a> Synthesizer<'a> {
    pub fn new(symbol_table: &'a SymbolTable) -> Self {
        Synthesizer { symbol_table }
    }

    pub fn synthesize(&self, flow_name: &str) -> Result<Vec<Context>, SynthesisError> {
        // Find the flow
        let flow = self.symbol_table.flows.get(flow_name)
            .ok_or(SynthesisError::FlowNotFound(flow_name.to_string()))?;
            
        let mut contexts = vec![Context::new()];
        
        for stmt in &flow.body {
            let mut next_contexts = Vec::new();
            
            for mut context in contexts {
                match stmt {
                    FlowStmt::Generator { target, source, constraints: _ } => {
                        // Check for Universe expansion (Direct Type usage: x <- TypeName)
                        if let Expr::Identifier(type_name) = source {
                             if let Some(ty_info) = self.symbol_table.types.get(type_name) {
                                 // It is a Type, so treat as Universe Source
                                 
                                 if let Some(components) = &ty_info.components {
                                     // Branching!
                                     for comp in components {
                                         let mut new_ctx = context.clone();
                                         
                                         // Remove heuristic. Use explicit structure from TypeInfo if available.
                                         let struct_type = if let Some(s) = &ty_info.structure {
                                             s.clone()
                                         } else {
                                             ty_info.parent.clone()
                                         };
                                         
                                         let node = ExecutionNode::Source {
                                             name: format!("Universe({})", comp), 
                                             type_name: struct_type.clone(),
                                         };
                                         let id = new_ctx.add_node(node);
                                         
                                         new_ctx.variables.insert(target.clone(), VariableState {
                                             name: target.clone(),
                                             type_name: struct_type,
                                             properties: ty_info.properties.clone(),
                                             node_id: id,
                                         });
                                         next_contexts.push(new_ctx);
                                     }
                                     continue;
                                 } else {
                                     // Single Universe Type
                                     let struct_type = if let Some(s) = &ty_info.structure {
                                         s.clone()
                                     } else {
                                         ty_info.parent.clone()
                                     };
                                     
                                     let node = ExecutionNode::Source {
                                         name: format!("Universe({})", type_name),
                                         type_name: struct_type.clone(),
                                     };
                                     let id = context.add_node(node);
                                     context.variables.insert(target.clone(), VariableState {
                                         name: target.clone(),
                                         type_name: struct_type, // Explicitly use the resolved structure type
                                         properties: ty_info.properties.clone(),
                                         node_id: id,
                                     });
                                     next_contexts.push(context);
                                     continue;
                                 }
                             }
                        }
                        
                        // Normal execution (no branching or not Universe)
                        match self.evaluate_expr(source, context) {
                            Ok(results) => {
                                for (mut ctx, type_name, props, node_id) in results {
                                    ctx.variables.insert(target.clone(), VariableState {
                                        name: target.clone(),
                                        type_name,
                                        properties: props,
                                        node_id,
                                    });
                                    next_contexts.push(ctx);
                                }
                            },
                            Err(e) => return Err(e),
                        }
                    },
                    FlowStmt::Assignment { target, expr } => {
                         match self.evaluate_expr(expr, context) {
                             Ok(results) => {
                                 for (mut ctx, type_name, props, node_id) in results {
                                     ctx.variables.insert(target.clone(), VariableState {
                                        name: target.clone(),
                                        type_name,
                                        properties: props,
                                        node_id,
                                    });
                                    next_contexts.push(ctx);
                                 }
                             },
                             Err(e) => return Err(e),
                         }
                    },
                    FlowStmt::Return(_) => {
                        // End of flow, keep context
                        next_contexts.push(context);
                    }
                }
            }
            contexts = next_contexts;
        }
        
        Ok(contexts)
    }

    pub fn evaluate_expr(&self, expr: &Expr, mut context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError> {
        match expr {
            Expr::Call { path, args } => {
                // Handle Universe logic
                if path.segments.last().unwrap() == "Universe" {
                    if let Some(Expr::Identifier(type_name)) = args.get(0) {
                        let ty_info = self.symbol_table.types.get(type_name)
                            .ok_or(SynthesisError::TypeMismatch("Unknown Type".into(), type_name.clone()))?;
                        
                        // Create Source node
                        let node = ExecutionNode::Source { 
                            name: format!("Universe({})", type_name), 
                            type_name: "DataFrame".to_string() // Fallback if called as Expr (should be handled in Stmt, but okay)
                        };
                        let id = context.add_node(node);
                        return Ok(vec![(context, "DataFrame".to_string(), ty_info.properties.clone(), id)]);
                    }
                }

                if path.segments.last().unwrap() == "Universe_Series" {
                     if let Some(Expr::Identifier(type_name)) = args.get(0) {
                        let ty_info = self.symbol_table.types.get(type_name)
                            .ok_or(SynthesisError::TypeMismatch("Unknown Type".into(), type_name.clone()))?;
                        
                        let node = ExecutionNode::Source { 
                            name: format!("Universe_Series({})", type_name), 
                            type_name: "TimeSeries".to_string() 
                        };
                        let id = context.add_node(node);
                        return Ok(vec![(context, "TimeSeries".to_string(), ty_info.properties.clone(), id)]);
                    }
                }
                
                let func_name = path.segments.last().unwrap();
                
                // Normalizer removed from here to allow generic dispatch

                
                // Function Handlers
                // 'divide' handler is required because the SymbolTable (HashMap) does not support function overloading.
                // Generic dispatch can't disambiguate multiple 'divide' definitions in stdlib.cm based on argument types.
                if func_name == "divide" {
                     use crate::comet::functions::FunctionHandler;
                    return crate::comet::functions::divide::Divide.handle(self, args, context);
                }
                
                // Other functions (update_when, apply_filter) are handled by Generic Dispatch below.

                // Generic Behavior / Function Dispatch
                // 1. Evaluate arguments first to get types
                // Since each arg evaluation can branch, we have a combinatorial explosion of argument contexts.
                // Simplified: Evaluate args sequentially across branched contexts.
                
                #[derive(Clone, Debug)]
                struct ArgResult {
                    node_id: usize,
                    type_name: String,
                    properties: Vec<String>,
                }
                
                let mut current_states: Vec<(Context, Vec<ArgResult>)> = vec![(context.clone(), Vec::new())];
                
                for arg in args {
                    let mut next_states = Vec::new();
                    for (ctx, mut prev_args) in current_states {
                         let res = self.evaluate_expr(arg, ctx)?; // branches
                         for (new_ctx, t, p, id) in res {
                             let mut args_list = prev_args.clone();
                             args_list.push(ArgResult { node_id: id, type_name: t, properties: p });
                             next_states.push((new_ctx, args_list));
                         }
                    }
                    current_states = next_states;
                }
                
                // Dispatch logic.
                let mut results = Vec::new();
                
                for (mut ctx, arg_results) in current_states {
                     // Look for Impls matching func_name
                     
                     let mut found = false;
                     for impl_info in &self.symbol_table.implementations {
                          if impl_info.behavior == *func_name {
                              let mut matched = true;
                              if let Some(c) = &impl_info.constraints {
                                  if let Expr::PropertyCheck { target, property } = c {
                                      if let Expr::Identifier(id) = target.as_ref() {
                                          if let Some(idx) = impl_info.args.iter().position(|x| x == id) {
                                               if let Some(arg_res) = arg_results.get(idx) {
                                                    if !arg_res.properties.contains(property) {
                                                        matched = false;
                                                    }
                                               }
                                          }
                                      }
                                  }
                              }
                              
                              if matched {
                                  found = true;
                                  
                                  // Inline the implementation body!
                                  let mut scope_ctx = ctx.clone();
                                  
                                  // Bind params to args
                                  for (i, param_name) in impl_info.args.iter().enumerate() {
                                      if let Some(arg_res) = arg_results.get(i) {
                                          scope_ctx.variables.insert(param_name.clone(), VariableState {
                                              name: param_name.clone(),
                                              type_name: arg_res.type_name.clone(),
                                              properties: arg_res.properties.clone(),
                                              node_id: arg_res.node_id,
                                          });
                                      }
                                  }
                                  
                                  // Synthesize body (Assume Return stmt for now)
                                  let mut block_contexts = vec![scope_ctx];
                                  let mut final_returns = Vec::new();
                                  
                                  for stmt in &impl_info.body {
                                      let mut next_block_ctx = Vec::new();
                                      for b_ctx in block_contexts {
                                           match stmt {
                                               crate::comet::ast::Stmt::Return(e) | 
                                               crate::comet::ast::Stmt::Flow(crate::comet::ast::FlowStmt::Return(e)) => {
                                                   if let Ok(res) = self.evaluate_expr(e, b_ctx) {
                                                       for (res_ctx, t, props, id) in res {
                                                           final_returns.push((res_ctx, t, props, id));
                                                       }
                                                   }
                                               },
                                               _ => {}
                                           }
                                      }
                                      block_contexts = next_block_ctx; 
                                  }
                                  
                                  // Apply ensures properties
                                  for (res_ctx, t, mut props, id) in final_returns {
                                      if let Some(ensures) = &impl_info.ensures {
                                           for prop in ensures {
                                               if !props.contains(prop) {
                                                   props.push(prop.clone());
                                               }
                                           }
                                      }
                                      results.push((res_ctx, t, props, id));
                                  }
                              }
                          }
                     }
                     
                     if !found {
                          // Try looking up in functions
                          if let Some(func_info) = self.symbol_table.functions.get(func_name) {
                               let mut matched = true;
                               // Constraints check
                               if let Some(vals) = &func_info.constraints {
                                   if let Expr::PropertyCheck { target, property } = vals {
                                       if let Expr::Identifier(id) = target.as_ref() {
                                           if let Some(idx) = func_info.params.iter().position(|p| p.name == *id) {
                                                if let Some(arg_res) = arg_results.get(idx) {
                                                     if !arg_res.properties.contains(&property) {
                                                         matched = false;
                                                     }
                                                }
                                           }
                                       }
                                   }
                               }
                               
                               if matched {
                                   found = true;
                                   let mut branch_ctx = ctx.clone();
                                   let arg_ids: Vec<usize> = arg_results.iter().map(|a| a.node_id).collect();
                                   
                                   let node = ExecutionNode::Operation {
                                       op: OperatorOp::FunctionCall(func_info.name.clone()),
                                       args: arg_ids,
                                   };
                                   let new_id = branch_ctx.add_node(node);
                                   
                                   let ret_type = func_info.return_type.clone();
                                   
                                   // Properties
                                   let mut ret_props = Vec::new(); // No heuristic

                                   if let Some(ensures) = &func_info.ensures {
                                       for prop in ensures {
                                           if !ret_props.contains(prop) {
                                               ret_props.push(prop.clone());
                                           }
                                       }
                                   }
                                   results.push((branch_ctx, ret_type, ret_props, new_id));
                               }
                          }
                     }
                }
                
                if results.is_empty() {
                     // No valid implementation/function found (or constraints failed).
                     // Prune this branch.
                     Ok(vec![]) 
                } else {
                     Ok(results)
                }
            },
            Expr::BinaryOp { left, op, right } => {
                self.evaluate_binary_op(left, op, right, context)
            },
            Expr::Identifier(ident) => {
                let (type_name, properties, node_id) = {
                    let var = context.variables.get(ident)
                        .ok_or(SynthesisError::VariableNotFound(ident.clone()))?;
                    (var.type_name.clone(), var.properties.clone(), var.node_id)
                };
                Ok(vec![(context, type_name, properties, node_id)])
            },
            Expr::Literal(lit) => {
                 let val_str = match lit {
                     crate::comet::ast::Literal::Float(f) => f.to_string(),
                     crate::comet::ast::Literal::Integer(i) => i.to_string(),
                     _ => "0".to_string(),
                 };
                 let node = ExecutionNode::Constant { value: val_str, type_name: "Float".to_string() };
                 let id = context.add_node(node);
                 Ok(vec![(context, "Float".to_string(), vec!["Constant".to_string()], id)])
            },
            _ => Ok(vec![(context, "Unknown".to_string(), vec![], 0)]),
        }
    }

    fn evaluate_binary_op(&self, left: &Expr, op: &crate::comet::ast::Op, right: &Expr, context: Context) -> Result<Vec<(Context, String, Vec<String>, usize)>, SynthesisError> {
        let left_results = self.evaluate_expr(left, context)?;
        
        let mut final_results = Vec::new();
        
        for (ctx, lhs_type, _, lhs_id) in left_results {
            let right_results = self.evaluate_expr(right, ctx)?;
            
            for (mut ctx2, rhs_type, _, rhs_id) in right_results {
                // Helper to check for "Constant" property
                let is_constant = |ty_name: &str| -> bool {
                    if let Some(info) = self.symbol_table.types.get(ty_name) {
                        return info.properties.iter().any(|p| p == "Constant");
                    }
                    false
                };
                
                let lhs_const = is_constant(&lhs_type);
                let rhs_const = is_constant(&rhs_type);

                match op {
                    crate::comet::ast::Op::Div => {
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
                             return Err(SynthesisError::TypeMismatch(format!("Compatible Division Types (LHS: {})", lhs_type), rhs_type));
                        }

                        // Add OpDivide to graph
                        let op_node = ExecutionNode::Operation {
                            op: OperatorOp::Divide,
                            args: vec![lhs_id, rhs_id],
                        };
                        let id = ctx2.add_node(op_node);
                        final_results.push((ctx2, res_type, vec![], id));
                    },
                    _ => {
                        final_results.push((ctx2, "UnknownOp".to_string(), vec![], 0));
                    }
                }
            }
        }
        
        Ok(final_results)
    }
}
