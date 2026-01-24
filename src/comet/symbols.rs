use std::collections::HashMap;
use crate::comet::ast::{Expr, Block, Ident};

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub types: HashMap<Ident, TypeInfo>,
    pub behaviors: HashMap<Ident, BehaviorInfo>,
    pub implementations: Vec<ImplInfo>, // List of impls, lookup by behavior + types
    pub functions: HashMap<Ident, FuncInfo>,
    pub flows: HashMap<Ident, FlowInfo>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            types: HashMap::new(),
            behaviors: HashMap::new(),
            implementations: Vec::new(),
            functions: HashMap::new(),
            flows: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub name: Ident,
    pub parent: Ident,
    pub properties: Vec<Ident>,
    pub components: Option<Vec<Ident>>,
    pub structure: Option<Ident>,
}

#[derive(Debug, Clone)]
pub struct BehaviorInfo {
    pub name: Ident,
    pub args: Vec<Ident>, // e.g. ["A", "B"] generic params
    pub return_type: Option<Ident>,
}

#[derive(Debug, Clone)]
pub struct ImplInfo {
    pub name: Ident, // "Ratio"
    pub behavior: Ident, // "Comparator"
    pub args: Vec<Ident>, // ["A", "B"] - these are essentially bound to the specific impl
    // But wait, in `Impl Ratio implements Comparator(a, b)`...
    // The `a` and `b` in `Comparator(a, b)` map to the arguments of the Behavior.
    // And usually they are typed?
    // In Comet `Impl ... for Series` or `Impl ... for (Series, Series)`?
    // Looking at `test_basic.cm`:
    // `Implementation Ratio implements Comparator(a, b) where b is NonZero`
    // It doesn't explicitly state the input types in the signature like `impl Comparator for Float`.
    // It seems Comet uses a "Constraint-based" or "Duck-typed" dispatch logic during Synthesis.
    // The context provides `x` and `y`.
    // `Comparator(x, y)` is called.
    // We check valid Impls.
    // `Ratio` takes `a`, `b`.
    // We bind `a=x`, `b=y`.
    // Check constraints: `b is NonZero`.
    // If pass, Ratio is a candidate.
    
    // So ImplInfo needs to store the constraint expression.
    pub constraints: Option<Expr>,
    pub ensures: Option<Vec<String>>,
    pub body: Vec<crate::comet::ast::Stmt>,
}

#[derive(Debug, Clone)]
pub struct FuncInfo {
    pub name: Ident,
    pub params: Vec<ParamInfo>,
    pub return_type: Ident,
    // constraints?
}

#[derive(Debug, Clone)]
pub struct ParamInfo {
    pub name: Ident,
    pub ty: String,
}

#[derive(Debug, Clone)]
pub struct FlowInfo {
    pub name: Ident,
    pub body: Vec<crate::comet::ast::FlowStmt>,
}
