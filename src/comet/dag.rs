use crate::comet::synthesis::RealExpr;
use crate::comet::ast::{Ident, Literal};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DagOp {
    Literal(Literal),
    Identifier(Ident),
    CallFn {
        func_name: Ident,
        args: Vec<(Option<Ident>, usize)>, // References to NodeIds
    }
}

#[derive(Debug, Clone)]
pub struct DagNode {
    pub id: usize,
    pub op: DagOp,
}

pub struct DagBuilder {
    pub nodes: Vec<DagNode>,
    pub hash_cons: HashMap<DagOp, usize>,
}

impl DagBuilder {
    pub fn new() -> Self {
        DagBuilder {
            nodes: Vec::new(),
            hash_cons: HashMap::new(),
        }
    }

    pub fn insert_op(&mut self, op: DagOp) -> usize {
        if let Some(&id) = self.hash_cons.get(&op) {
            return id;
        }

        let new_id = self.nodes.len();
        self.nodes.push(DagNode { id: new_id, op: op.clone() });
        self.hash_cons.insert(op, new_id);
        new_id
    }

    /// Converts a RealExpr tree into a DAG. Recursively processes children.
    pub fn build_from_real_expr(&mut self, expr: &RealExpr) -> usize {
        match expr {
            RealExpr::Literal(lit) => {
                self.insert_op(DagOp::Literal(lit.clone()))
            }
            RealExpr::Identifier(ident) => {
                self.insert_op(DagOp::Identifier(ident.clone()))
            }
            RealExpr::CallFn { func_name, args, .. } => {
                let mut dag_args = Vec::new();
                for (name_opt, arg_expr) in args {
                    let child_id = self.build_from_real_expr(arg_expr);
                    dag_args.push((name_opt.clone(), child_id));
                }
                self.insert_op(DagOp::CallFn {
                    func_name: func_name.clone(),
                    args: dag_args,
                })
            }
        }
    }

    /// Converts a forest (Vec of RealExprs representing the independent roots like Behavior output + side effects)
    /// into a DAG and returns the DAG nodes and the roots.
    pub fn build_from_forest(&mut self, forest: &[RealExpr]) -> Vec<usize> {
        let mut root_ids = Vec::new();
        for tree in forest {
            root_ids.push(self.build_from_real_expr(tree));
        }
        root_ids
    }
}
