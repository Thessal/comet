use crate::comet::ast::{Ident, Program, Declaration, FlowDecl, Expr, FlowStmt, Literal, ConstraintDecl, Op};

/// The "Real AST" represents a fully synthesized and unrolled execution graph.
/// All behaviors are strictly mapped to their concrete `Fn` implementations.
#[derive(Debug, Clone)]
pub struct RealProgram {
    pub flows: Vec<RealFlow>,
}

/// A Flow containing all valid Cartesian product combinations of its execution graph.
#[derive(Debug, Clone)]
pub struct RealFlow {
    pub name: Ident,
    pub combinations: Vec<RealFlowImpl>,
}

/// A single, valid exact mapping of a Flow into specific `Fn`s.
#[derive(Debug, Clone)]
pub struct RealFlowImpl {
    pub stmts: Vec<RealStmt>,
}

#[derive(Debug, Clone)]
pub enum RealStmt {
    Assignment(Ident, RealExpr),
    Expr(RealExpr),
}

#[derive(Debug, Clone)]
pub enum RealExpr {
    CallFn {
        func_name: Ident, // Strictly a concrete `Fn`
        args: Vec<(Option<Ident>, RealExpr)>,
        return_constraint: ConstraintDecl, // The exactly resolved output constraint
    },
    Literal(Literal),
    Identifier(Ident),
    BinaryOp { left: Box<RealExpr>, op: Op, right: Box<RealExpr> },
    UnaryOp { op: Op, target: Box<RealExpr> },
}

pub struct Synthesizer<'a> {
    pub ast: &'a Program,
    // Symbol tables can be built from ast.declarations
}

impl<'a> Synthesizer<'a> {
    pub fn new(ast: &'a Program) -> Self {
        Synthesizer { ast }
    }

    pub fn synthesize(&self) -> Result<RealProgram, String> {
        let mut real_program = RealProgram { flows: Vec::new() };

        for decl in &self.ast.declarations {
            if let Declaration::Flow(flow_decl) = decl {
                let real_flow = self.synthesize_flow(flow_decl)?;
                real_program.flows.push(real_flow);
            }
        }

        Ok(real_program)
    }

    fn synthesize_flow(&self, flow: &FlowDecl) -> Result<RealFlow, String> {
        // Start with a single empty implementation
        let mut combinations = vec![RealFlowImpl { stmts: Vec::new() }];

        for stmt in &flow.body {
            // Each statement might expand into multiple possibilities
            let mut next_combinations = Vec::new();

            for current_impl in combinations {
                // Evaluate the statement into [RealStmt possibilities]
                let possible_stmts = self.synthesize_stmt(stmt, &current_impl)?;

                // Cartesian product: split the current branch
                for real_stmt in possible_stmts {
                    let mut cloned_impl = current_impl.clone();
                    cloned_impl.stmts.push(real_stmt);
                    next_combinations.push(cloned_impl);
                }
            }
            combinations = next_combinations;
        }

        Ok(RealFlow {
            name: flow.name.clone(),
            combinations,
        })
    }

    fn synthesize_stmt(&self, stmt: &FlowStmt, context: &RealFlowImpl) -> Result<Vec<RealStmt>, String> {
        match stmt {
            FlowStmt::Assignment { target, expr } => {
                let expr_combinations = self.synthesize_expr(expr, context)?;
                Ok(expr_combinations.into_iter()
                    .map(|e| RealStmt::Assignment(target.clone(), e))
                    .collect())
            },
            FlowStmt::Expr(expr) => {
                let expr_combinations = self.synthesize_expr(expr, context)?;
                Ok(expr_combinations.into_iter()
                    .map(RealStmt::Expr)
                    .collect())
            }
        }
    }

    fn synthesize_expr(&self, expr: &Expr, context: &RealFlowImpl) -> Result<Vec<RealExpr>, String> {
        match expr {
            Expr::Literal(lit) => Ok(vec![RealExpr::Literal(lit.clone())]),
            Expr::Identifier(ident) => Ok(vec![RealExpr::Identifier(ident.clone())]),
            Expr::Call { path, args } => {
                let target_name = path.segments.last().unwrap();
                
                // 1. Resolve arguments first (Cartesian product of all argument combinations)
                let resolved_args = self.resolve_args(args, context)?;
                
                // 2. Identify if target_name is a Function or Behavior
                // TODO: Look up target_name in Symbol Table
                // if `behavior`, find all valid `Fn`s that map to this behavior based on `resolved_args` constraints
                // if `Fn`, return that directly.
                
                // For demonstration, we simply map it dynamically to a `todo!()` abstraction
                let mut possibilities = Vec::new();
                for resolved_arg_list in resolved_args {
                    // TODO: Resolve constraint unification (e.g., tying `'a` across input/output)
                    // TODO: Pattern matching to drop mismatched `Fn` implementations
                    
                    // Stub: Just returning the target_name as a CallFn for prototype
                    possibilities.push(RealExpr::CallFn {
                        func_name: target_name.clone(), // This should be replaced with matched `Fn`(s)
                        args: resolved_arg_list,
                        // This constraint is a placeholder
                        return_constraint: ConstraintDecl {
                            base_type: crate::comet::ast::TypeDecl::DataFrame,
                            category_expr: None,
                        }
                    });
                }
                Ok(possibilities)
            },
            Expr::BinaryOp { left, op, right } => {
                let left_combs = self.synthesize_expr(left, context)?;
                let right_combs = self.synthesize_expr(right, context)?;
                
                let mut res = Vec::new();
                for l in &left_combs {
                    for r in &right_combs {
                        res.push(RealExpr::BinaryOp { left: Box::new(l.clone()), op: op.clone(), right: Box::new(r.clone()) });
                    }
                }
                Ok(res)
            },
            Expr::UnaryOp { op, target } => {
                let tgt_combs = self.synthesize_expr(target, context)?;
                Ok(tgt_combs.into_iter().map(|t| RealExpr::UnaryOp { op: op.clone(), target: Box::new(t) }).collect())
            }
            _ => Err("Unsupported or incomplete Expr mapping in RealAST Synthesis".into()),
        }
    }
    
    fn resolve_args(&self, args: &Vec<crate::comet::ast::ArgValue>, context: &RealFlowImpl) -> Result<Vec<Vec<(Option<Ident>, RealExpr)>>, String> {
        let mut arg_combinations: Vec<Vec<(Option<Ident>, RealExpr)>> = vec![vec![]];
        
        for arg in args {
            let expr_combs = self.synthesize_expr(&arg.value, context)?;
            let mut next_combs = Vec::new();
            
            for base_list in arg_combinations {
                for expr in &expr_combs {
                    let mut new_list = base_list.clone();
                    new_list.push((arg.name.clone(), expr.clone()));
                    next_combs.push(new_list);
                }
            }
            arg_combinations = next_combs;
        }
        
        Ok(arg_combinations)
    }
}
