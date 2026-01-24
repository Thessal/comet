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

#[derive(Debug, Clone)]
pub struct Context {
    // Map variable name to its "Type" or "Semantic State"
    // In Comet, a variable has a Structural Type (DataFrame) and a set of Semantic Properties (Monetary, NonZero).
    // We can represent this as a "Rich Type".
    pub variables: HashMap<Ident, VariableState>,
}

impl Context {
    pub fn new() -> Self {
        Context {
            variables: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariableState {
    pub name: Ident,
    pub type_name: String, // e.g. "PERatio", "DataFrame"
    pub properties: Vec<String>, // e.g. ["Monetary", "Ranged"]
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
            
        let mut context = Context::new();
        
        for stmt in &flow.body {
            match stmt {
                FlowStmt::Generator { target, source, constraints: _ } => {
                    // Evaluate source (e.g. Universe(PERatio))
                    let (type_name, props) = self.evaluate_expr(source, &context)?;
                    context.variables.insert(target.clone(), VariableState {
                        name: target.clone(),
                        type_name,
                        properties: props,
                    });
                },
                FlowStmt::Assignment { target, expr, constraints: _ } => {
                    let (type_name, props) = self.evaluate_expr(expr, &context)?;
                     context.variables.insert(target.clone(), VariableState {
                        name: target.clone(),
                        type_name,
                        properties: props,
                    });
                },
                FlowStmt::Return(_) => {
                    // End of flow
                }
            }
        }
        
        Ok(vec![context])
    }

    pub fn evaluate_expr(&self, expr: &Expr, context: &Context) -> Result<(String, Vec<String>), SynthesisError> {
        match expr {
            Expr::Call { path, args } => {
                // Handle Universe logic
                if path.segments.last().unwrap() == "Universe" {
                    if let Some(Expr::Identifier(type_name)) = args.get(0) {
                        let ty_info = self.symbol_table.types.get(type_name)
                            .ok_or(SynthesisError::TypeMismatch("Unknown Type".into(), type_name.clone()))?;
                        return Ok(("DataFrame".to_string(), ty_info.properties.clone()));
                    }
                }

                if path.segments.last().unwrap() == "Universe_Series" {
                     if let Some(Expr::Identifier(type_name)) = args.get(0) {
                        let ty_info = self.symbol_table.types.get(type_name)
                            .ok_or(SynthesisError::TypeMismatch("Unknown Type".into(), type_name.clone()))?;
                        return Ok(("TimeSeries".to_string(), ty_info.properties.clone()));
                    }
                }
                
                let func_name = path.segments.last().unwrap();
                
                // Dispatch to Behavior Handlers
                // specific name check for now since we don't have dynamic loading yet
                if func_name == "Normalizer" {
                    use crate::comet::behaviors::BehaviorHandler;
                    return crate::comet::behaviors::normalizer::Normalizer.handle(self, args, context);
                }
                
                // Dispatch to Function Handlers
                if func_name == "update_when" {
                    use crate::comet::functions::FunctionHandler;
                    return crate::comet::functions::update_when::UpdateWhen.handle(self, args, context);
                }

                if func_name == "apply_filter" {
                    // Logic: returns arg0 type + "Masked" property
                   if let Some(arg0) = args.get(0) {
                       let (t, mut props) = self.evaluate_expr(arg0, context)?;
                       props.push("Masked".to_string());
                       return Ok((t, props));
                   }
                }

                // Default recursion
                 if let Some(arg0) = args.get(0) {
                     return self.evaluate_expr(arg0, context);
                 }
                
                Ok(("Unknown".to_string(), vec![]))
            },
            Expr::BinaryOp { left, op, right } => {
                self.evaluate_binary_op(left, op, right, context)
            },
            Expr::Identifier(ident) => {
                let var = context.variables.get(ident)
                    .ok_or(SynthesisError::VariableNotFound(ident.clone()))?;
                Ok((var.type_name.clone(), var.properties.clone()))
            },
            _ => Ok(("Unknown".to_string(), vec![])),
        }
    }

    fn evaluate_binary_op(&self, left: &Expr, op: &crate::comet::ast::Op, right: &Expr, context: &Context) -> Result<(String, Vec<String>), SynthesisError> {
        let (lhs_type, _) = self.evaluate_expr(left, context)?;
        let (rhs_type, _) = self.evaluate_expr(right, context)?;
        
        // Helper to check for "Constant" property
        let is_constant = |ty_name: &str| -> bool {
            if let Some(info) = self.symbol_table.types.get(ty_name) {
                return info.properties.iter().any(|p| p == "Constant");
            }
            false
        };

        match op {
            crate::comet::ast::Op::Div => {
                // Type Matrix:
                // DF / DF -> DF
                // DF / TS -> DF
                // DF / Const -> DF
                // TS / TS -> TS
                // TS / Const -> TS
                // Const / Const -> Const  (New Requirement)
                
                let lhs_const = is_constant(&lhs_type);
                let rhs_const = is_constant(&rhs_type);

                if lhs_type == "DataFrame" {
                    if rhs_type == "DataFrame" || rhs_type == "TimeSeries" || rhs_const {
                       return Ok(("DataFrame".to_string(), vec![]));
                    }
                } else if lhs_type == "TimeSeries" {
                    if rhs_type == "TimeSeries" || rhs_const {
                        return Ok(("TimeSeries".to_string(), vec![]));
                    }
                } else if lhs_const {
                    if rhs_const {
                         // Attempt to find generic Constant type or inherit? 
                         // For now, return Float or assume LHS type if it is a constant type (like Float)
                         return Ok((lhs_type, vec![])); 
                    }
                }
                
                Err(SynthesisError::TypeMismatch(format!("Compatible Division Types (LHS: {})", lhs_type), rhs_type))
            },
            _ => Ok(("UnknownOp".to_string(), vec![])),
        }
    }
}
