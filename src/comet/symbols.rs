use std::collections::HashMap;
use crate::comet::ast::{Expr, Block, Ident, Constraint, TypedArg};

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub types: HashMap<Ident, TypeInfo>,
    pub behaviors: HashMap<Ident, BehaviorInfo>,
    pub functions: HashMap<Ident, FuncInfo>,
    pub flows: HashMap<Ident, FlowInfo>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            types: HashMap::new(),
            behaviors: HashMap::new(),
            functions: HashMap::new(),
            flows: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: Ident,
    pub parent_constraint: Option<Constraint>,
    pub properties: Vec<Ident>,
    pub components: Option<Vec<Ident>>,
    pub structure: Option<Ident>, // Keeping for now if AST still has it in StructDecl, but TypeDecl removed it.
}

#[derive(Debug, Clone)]
pub struct BehaviorInfo {
    pub name: Ident,
    pub args: Vec<TypedArg>, 
    pub return_type: Constraint,
}

#[derive(Debug, Clone)]
pub struct FuncInfo {
    pub name: Ident,
    pub params: Vec<TypedArg>,
    pub return_type: Constraint,
    pub body: crate::comet::ast::Block,
}

#[derive(Debug, Clone)]
pub struct FlowInfo {
    pub name: Ident,
    pub body: Vec<crate::comet::ast::FlowStmt>,
}
