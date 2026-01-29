use std::collections::{HashMap, HashSet};
use crate::comet::ast::{FlowStmt, Expr, Ident, Stmt, Constraint};
use crate::comet::symbols::{SymbolTable};
use crate::comet::constraints::{expand, matches_chain, Atom, ConstraintSet};
use thiserror::Error;
use crate::comet::ir::{ExecutionGraph, ExecutionNode, OperatorOp};

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
    #[error("Synthesis Error: {0}")]
    SynthesisError(String),
}

#[derive(Debug, Clone)]
pub struct Context {
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
    // The set of possible Type Chains this variable currently holds.
    // e.g. { [Series, NonZero], [DataFrame] }
    pub constraint_set: ConstraintSet,
    pub node_id: usize,
}

pub struct Synthesizer<'a> {
    pub symbol_table: &'a SymbolTable,
}

#[derive(Clone, Debug)]
pub struct ArgResult {
    pub node_id: usize,
    pub constraint_set: ConstraintSet,
    pub name: Option<Ident>,
}

impl<'a> Synthesizer<'a> {
    pub fn new(symbol_table: &'a SymbolTable) -> Self {
        Synthesizer { symbol_table }
    }

    pub fn synthesize(&self, flow_name: &str) -> Result<Vec<Context>, SynthesisError> {
        let flow = self.symbol_table.flows.get(flow_name)
            .ok_or(SynthesisError::FlowNotFound(flow_name.to_string()))?;
            
        let mut contexts = vec![Context::new()];
        
        for stmt in &flow.body {
            let mut next_contexts = Vec::new();
            
            for context in contexts {
                match stmt {
                    FlowStmt::Assignment { target, expr } => {
                        match self.evaluate_expr(expr, context) {
                             Ok(results) => {
                                 for (mut ctx, c_set, node_id) in results {
                                     ctx.variables.insert(target.clone(), VariableState {
                                        name: target.clone(),
                                        constraint_set: c_set,
                                        node_id,
                                    });
                                    next_contexts.push(ctx);
                                 }
                             },
                             Err(e) => return Err(e),
                         }
                    },
                    FlowStmt::Return(_) => { // TODO: Handle return
                        next_contexts.push(context);
                    },
                    // Function Handlers
                }
            }
            contexts = next_contexts;
        }
        
        Ok(contexts)
    }

    pub fn evaluate_expr(&self, expr: &Expr, mut context: Context) -> Result<Vec<(Context, ConstraintSet, usize)>, SynthesisError> {
        match expr {
            Expr::Call { path, args } => {
                let func_name = path.segments.last().unwrap();
                
                // Evaluate all arguments first
                // Branching possible
                
                // Generic Behavior / Function Dispatch
                // 1. Evaluate arguments first to get types
                
                
                let mut current_states: Vec<(Context, Vec<ArgResult>)> = vec![(context.clone(), Vec::new())];
                
                for arg in args {
                    let mut next_states = Vec::new();
                    for (ctx, prev_args) in current_states {
                         let res = self.evaluate_expr(&arg.value, ctx)?; 
                         for (new_ctx, c_set, id) in res {
                             let mut args_list = prev_args.clone();
                             args_list.push(ArgResult { node_id: id, constraint_set: c_set, name: arg.name.clone() });
                             next_states.push((new_ctx, args_list));
                         }
                    }
                    current_states = next_states;
                }
                
                // Match Behavior/Function logic
                let mut results = Vec::new();
                let state_count = current_states.len();

                for (mut ctx, arg_results) in current_states {
                     // 1. Try to find a Behavior that matches
                     // If it is a Behavior, we want to EXPAND it into all possible implementations.
                     // Implementations can be:
                     // A) Concrete Functions that match the signature
                     // B) Variants defined in the Behavior's return type (e.g. "21" | "63")
                     
                     let mut found = false;

                     // Check if it is a Behavior
                     if let Some(beh_info) = self.symbol_table.behaviors.get(func_name) {
                         if self.check_args_match(&beh_info.args, &arg_results) {
                             found = true;
                             let arg_ids: Vec<usize> = arg_results.iter().map(|a| a.node_id).collect();

                             // A) Find Matching Functions
                             for (fn_name, fn_info) in &self.symbol_table.functions {
                                 // Check 1: Return type must be compatible
                                 // The function return type (fn_info.return_type) must satisfy the behavior requirement (beh_info.return_type).
                                 // We use `expand` to get the atoms and check inclusion.
                                 // Note: Function return might be narrower (more specific). 
                                 // e.g. Behavior returns "DataFrame", Function returns "DataFrame Volume". This is OK.
                                 // e.g. Behavior returns "DataFrame Volume", Function returns "DataFrame". This is NOT OK.
                                 // So we check: behavior_constraints \subseteq function_constraints? NO.
                                 // Function provides {A, B}. Behavior requires {A}. {A} \subseteq {A, B}.
                                 // So Behavior constraints must be a subset of Function constraints.
                                 
                                 let beh_constraints = expand(&beh_info.return_type);
                                 let fn_constraints = expand(&fn_info.return_type);
                                 
                                 // Check if ALL behavior constraints are satisfied by function
                                 // Since expanded constraints are usually sets of chains, this can be complex.
                                 // Simplified: Check if behavior constraint chain matches function constraint chain.
                                 // Check args first as it's cheaper.
                                 
                                 if self.check_args_match(&fn_info.params, &arg_results) {
                                     let fn_constraints = expand(&fn_info.return_type);
                                 
                                     // Fully expand function constraints to include inherited properties
                                     let mut full_fn_constraints = HashSet::new();
                                     for chain in fn_constraints {
                                         full_fn_constraints.insert(self.fully_expand_chain(chain));
                                     }

                                     // Check if ALL fn chains satisfy behavior requirement
                                     let mut compatible = true;
                                     if full_fn_constraints.is_empty() {
                                         compatible = false;
                                     } 
                                     for f_chain in &full_fn_constraints {
                                         if !matches_chain(f_chain, &beh_info.return_type) {
                                             compatible = false;
                                             break;
                                         }
                                     }
                                     
                                     if compatible {
                                         let node = ExecutionNode::Operation {
                                             op: OperatorOp::FunctionCall(fn_name.clone()),
                                             args: arg_ids.clone(),
                                         };
                                         let new_id = ctx.add_node(node);
                                         
                                         // Return the Expanded constraints? Or original?
                                         // Usually we want the Node to carry Full info.
                                         results.push((ctx.clone(), full_fn_constraints, new_id));
                                     }
                                 }
                             }

                             // B) Extract Variants (Literals) from Behavior Return Type
                             // e.g. "21" | "63"
                             let variants = self.collect_variants(&beh_info.return_type);
                             for variant in variants {
                                 // Create a Constant node for the variant
                                 // If variant looks like number -> Constant. If string -> Source(Universe)?
                                 // For now treat as Constant string.
                                 let node = ExecutionNode::Constant {
                                     value: variant.clone(),
                                     type_name: "Constant".to_string(), // TODO: Infer type?
                                 };
                                 let new_id = ctx.add_node(node);
                                 let mut set = HashSet::new();
                                 let base_constraints = expand(&beh_info.return_type);
                                 for chain in base_constraints {
                                     if chain.contains(&Atom::Type(variant.clone())) {
                                         set.insert(chain);
                                     }
                                 }
                                 if set.is_empty() {
                                     set.insert(vec![Atom::Type(variant.clone())]);
                                 }
                                 results.push((ctx.clone(), set, new_id));
                             }
                             
                             // If NO implementations found (function or variant), fallback to Abstract Behavior Node?
                             // Prompt implies we should define "combinations". Matches "all possible pattern".
                             // If we have 0 concrete implementations, synthesis stops (dead end).
                             // Previously we emitted an abstract node.
                             // Keep emitting abstract node ONLY if results is empty? 
                             // Or always? 
                             // If we want to "expand", we prefer concrete.
                             // If we have concrete, do we still include abstract? Usually NO.
                             
                             /*
                             if results.is_empty() {
                                  // Fallback to abstract
                                  let node = ExecutionNode::Operation {
                                      op: OperatorOp::FunctionCall(beh_info.name.clone()),
                                      args: arg_ids,
                                  };
                                  let new_id = ctx.add_node(node);
                                  let ret_constraints = expand(&beh_info.return_type);
                                  results.push((ctx.clone(), ret_constraints, new_id));
                             }
                             */
                         }
                     } 
                     
                     // 2. Exact Function Match (if func_name is a concrete function)
                     // If it was valid behavior, we expanded it. But `func_name` might NOT be behavior.
                     // Or it might be BOTH? (Shadowing).
                     // If we expanded as Behavior, `found` is true.
                     
                     if !found {
                         if let Some(func_info) = self.symbol_table.functions.get(func_name) {
                             if self.check_args_match(&func_info.params, &arg_results) {
                                 found = true;
                                 let arg_ids: Vec<usize> = arg_results.iter().map(|a| a.node_id).collect();
                                 let node = ExecutionNode::Operation {
                                     op: OperatorOp::FunctionCall(func_info.name.clone()),
                                     args: arg_ids,
                                 };
                                 let new_id = ctx.add_node(node);
                                 let ret_constraints = expand(&func_info.return_type);
                                 results.push((ctx.clone(), ret_constraints, new_id));
                             }
                         }
                     }
                     
                     // 3. Flows
                     if !found {
                          if let Some(_) = self.symbol_table.flows.get(func_name) {
                             if args.is_empty() {
                                 found = true;
                                 let flow_contexts = self.synthesize(func_name)?;
                                 // Cartesian product if flow returns multiple contexts?
                                 // `flow_contexts` is Vec<Context>. Each context has "result".
                                 // We need to merge EACH flow context into CURRENT context.
                                 // Merging ExecutionGraphs is non-trivial if they share history.
                                 // But here `Context` is immutable snapshot.
                                 // We can treat the flow result as a Source in our current graph?
                                 // Or simpler: Just take the result variable properties.
                                 // But `flow_contexts` might represent 32 DIFFERENT ways to compute result.
                                 // We should branch our current context 32 times.
                                 
                                 for flow_ctx in flow_contexts {
                                     if let Some(res_var) = flow_ctx.variables.get("result") {
                                          let node = ExecutionNode::Source { 
                                             name: format!("FlowResult({}:{})", func_name, res_var.node_id),
                                             type_name: "FlowReference".to_string() 
                                          };
                                          // Note: We are losing the actual graph of the flow here. 
                                          // In full synthesis we would flatten/inline.
                                          // For counting, this is fine.
                                          let id = ctx.add_node(node);
                                          results.push((ctx.clone(), res_var.constraint_set.clone(), id));
                                     }
                                 }
                             }
                          }
                     }
                     
                     if !found && results.is_empty() {
                          // Try builtins or error
                     }
                }
                
                if results.is_empty() {
                    return Err(SynthesisError::NoImplFound(func_name.to_string()));
                }
                
                Ok(results)
            },
            Expr::Identifier(ident) => {
                let (c_set, node_id) = {
                    let var = context.variables.get(ident)
                        // .ok_or(SynthesisError::VariableNotFound(ident.clone()))?;
                        ;
                    if let Some(v) = var {
                        (v.constraint_set.clone(), v.node_id)
                    } else {
                        // Check if it is a Type name (Source/Generator)
                        // In new spec/ast, we might use calls for generators, but if standard "Earnings" is identifier:
                        if let Some(ty_info) = self.symbol_table.types.get(ident) {
                             // It is a Type. Create a Universe Source for this type.
                             let node = ExecutionNode::Source { 
                                 name: format!("Universe({})", ident), 
                                 type_name: ident.clone() 
                             };
                             let id = context.add_node(node);
                             
                             // Construct ConstraintSet from Type properties
                             // Type A : B C { P, Q }
                             // The type A implies it HAS properties P, Q, and inherits from B (and C).
                             // We need to resolve all atomic properties.
                             // Simplified: Just add the Type name itself as Atom?
                             // And add properties as Atoms.
                             
                             let mut set = HashSet::new();
                             let mut chain = vec![Atom::Type(ident.clone())];
                             for p in &ty_info.properties {
                                 chain.push(Atom::Type(p.clone()));
                             }
                             // TODO: Recurse parent
                             set.insert(chain);
                             
                             (set, id)
                         } else if let Some(flow_info) = self.symbol_table.flows.get(ident) { 
                             // It is a Flow. 
                             // We should Synthesize this flow and use its result.
                             // NOTE: This assumes recursiveness is limited or handled? 
                             // For now, simple recursive synthesis.
                             
                             // We need a way to reuse results if already synthesized in `main.rs`.
                             // But here we might just re-synthesize or assume it's external.
                             // Since `evaluate_expr` returns `Context` (updated), we should be careful.
                             // Actually, if we synthesize another flow, it has its OWN context.
                             // Does it return a Value that we can use? 
                             // `synthesize` returns `Vec<Context>`.
                             // We need the "Return Value" of that flow.
                             // Flows implicit return is the last assignment or `result` variable?
                             // Let's assume flow result is in variable named "result" (from parser) or similar?
                             // Or just the LAST node added?
                             // `spec_test_v2.cm` says `flow volume_spike = ...`
                             // Parser converts to `result = ...`
                             // So the variable "result" in the final context of `volume_spike` holds the flow output.
                             
                             let flow_contexts = self.synthesize(ident)?;
                             // Take the first valid context (assuming 1 path for now)
                             if let Some(final_ctx) = flow_contexts.first() {
                                 // Look for "result" variable
                                 if let Some(res_var) = final_ctx.variables.get("result") {
                                      // We need to IMPORT this node/variable into CURRENT context.
                                      // Node ID in `final_ctx` is local to it.
                                      // We might need to map it or treat it as an external reference.
                                      // Simplest hack: Clone the constraint set and create a "FlowRef" node in current context.
                                      
                                      let node = ExecutionNode::Source { 
                                         name: format!("FlowResult({})", ident),
                                         type_name: "FlowRefernce".to_string() // Todo: real type?
                                      };
                                      let id = context.add_node(node);
                                      
                                      (res_var.constraint_set.clone(), id)
                                 } else {
                                     // Flow didn't assign result?
                                      return Err(SynthesisError::ConstraintFailed(format!("Flow {} did not return a result", ident)));
                                 }
                             } else {
                                  return Err(SynthesisError::SynthesisError(format!("Flow {} failed to synthesize", ident)));
                             }
                         } else {
                             return Err(SynthesisError::VariableNotFound(ident.clone()));
                         }
                    }
                };
                Ok(vec![(context, c_set, node_id)])
            },
            Expr::Literal(lit) => {
                 let val_str = match lit {
                     crate::comet::ast::Literal::Float(f) => f.to_string(),
                     crate::comet::ast::Literal::Integer(i) => i.to_string(),
                     _ => "0".to_string(),
                 };
                 let node = ExecutionNode::Constant { value: val_str, type_name: "Constant".to_string() };
                 let id = context.add_node(node);
                 
                 let mut set = HashSet::new();
                 set.insert(vec![Atom::Type("Constant".to_string())]);
                 
                 Ok(vec![(context, set, id)])
            },
            Expr::BinaryOp { left, op, right } => {
                // Implement simple type check for div/mul if needed
                // For now, assume Result = Union of inputs? No.
                // Binary Ops usually result in a new Type.
                // Assuming internal implementation handles this or they are valid Impls/Funcs?
                // Spec says: `return dividend / divisor` inside a function body.
                // We are synthesizing FLOWS. Flows call Functions.
                // If a Flow has BinaryOp, it's syntax sugar?
                // `flow x = a / b` -> `flow x = Div(a, b)`.
                // Let's assume Map to "div", "mul", "add", "sub" behaviors.
                
                let func_name = match op {
                    crate::comet::ast::Op::Div => "divide", // Standard library should have `behavior divide`
                    crate::comet::ast::Op::Mul => "multiply",
                    crate::comet::ast::Op::Add => "add",
                    crate::comet::ast::Op::Sub => "subtract",
                    _ => "unknown_op",
                };
                
                let call_expr = Expr::Call { 
                    path: crate::comet::ast::Path { segments: vec![func_name.to_string()] },
                    args: vec![
                        crate::comet::ast::ArgValue { name: None, value: *left.clone() },
                        crate::comet::ast::ArgValue { name: None, value: *right.clone() }
                    ] 
                };
                
                self.evaluate_expr(&call_expr, context)
            },
            _ => Ok(vec![]),
        }
    }

    fn fully_expand_chain(&self, chain: Vec<Atom>) -> Vec<Atom> {
        let mut full_chain = chain.clone();
        let mut visited = HashSet::new();
        
        let mut stack = Vec::new();
        for atom in &chain {
            if let Atom::Type(name) = atom {
                stack.push(name.clone());
            }
        }
        
        while let Some(ty_name) = stack.pop() {
            if visited.contains(&ty_name) { continue; }
            visited.insert(ty_name.clone());
            
            if let Some(ty_info) = self.symbol_table.types.get(&ty_name) {
                for prop in &ty_info.properties {
                    let atom = Atom::Type(prop.clone());
                    if !full_chain.contains(&atom) {
                        full_chain.push(atom);
                        stack.push(prop.clone());
                    }
                }
                if let Some(parent_c) = &ty_info.parent_constraint {
                     let parent_chains = expand(parent_c);
                     for p_chain in parent_chains {
                         for atom in p_chain {
                             if !full_chain.contains(&atom) {
                                  full_chain.push(atom.clone());
                                  if let Atom::Type(name) = atom {
                                      stack.push(name);
                                  }
                             }
                         }
                     }
                }
            }
        }
        full_chain.sort_by(|a, b| match (a, b) {
             (Atom::Type(s1), Atom::Type(s2)) => s1.cmp(s2),
             (Atom::Variable(s1), Atom::Variable(s2)) => s1.cmp(s2),
             (Atom::Type(_), Atom::Variable(_)) => std::cmp::Ordering::Less,
             (Atom::Variable(_), Atom::Type(_)) => std::cmp::Ordering::Greater,
        });
        full_chain
    }

    fn collect_variants(&self, constraint: &Constraint) -> Vec<String> {
        match constraint {
            Constraint::Atom(name) => {
                // Heuristic: If it is NOT a known Type, it is a variant.
                if !self.symbol_table.types.contains_key(name) {
                     vec![name.clone()]
                } else {
                     vec![]
                }
            },
            Constraint::Union(cs) => {
                let mut vars = Vec::new();
                for c in cs {
                    vars.extend(self.collect_variants(c));
                }
                vars
            },
            Constraint::Addition(cs) => {
                let mut vars = Vec::new();
                for c in cs {
                    vars.extend(self.collect_variants(c));
                }
                vars
            },
             _ => vec![],
        }
    }
    
    fn check_args_match(&self, required: &Vec<crate::comet::ast::TypedArg>, provided: &Vec<ArgResult>) -> bool {
        if required.len() != provided.len() {
            return false;
        }
        
        // Reorder provided args to match required args
        let mut ordered_provided = Vec::new();
        
        // 1. Map name to provided arg
        let mut name_map = HashMap::new();
        let mut positionals = Vec::new();
        
        for p in provided {
            if let Some(n) = &p.name {
                name_map.insert(n.clone(), p);
            } else {
                positionals.push(p);
            }
        }
        
        let mut pos_idx = 0;
        
        for req in required {
            let matched_arg = if let Some(arg) = name_map.remove(&req.name) {
                arg
            } else {
                if pos_idx < positionals.len() {
                    let arg = positionals[pos_idx];
                    pos_idx += 1;
                    arg
                } else {
                    return false; // Missing argument
                }
            };
            ordered_provided.push(matched_arg);
        }
        
        if pos_idx != positionals.len() {
             return false; // Extra positional arguments
        }
        
        if !name_map.is_empty() {
            return false; // Extra named arguments
        }
        
        for (req, prov) in required.iter().zip(ordered_provided.into_iter()) {
            
            for chain in &prov.constraint_set {
                if !matches_chain(chain, &req.constraint) {
                     // println!("Debug: Match failed for arg '{}'. Req: {:?}, Prov Chain: {:?}", req.name, req.constraint, chain);
                    return false;
                }
            }
        }
        true
    }
}

// Inner struct for ArgResult is defined inside method, might need to be moved out if reused.
// For now checks are inline.
