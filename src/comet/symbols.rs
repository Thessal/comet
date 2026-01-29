use std::collections::HashMap;
use crate::comet::ast::{Expr, Ident, TypeRef, AdtDecl, Constructor, FuncDecl, Constraint};

#[derive(Debug, Clone)]
pub struct SymbolTable {
    pub adts: HashMap<Ident, AdtInfo>,
    pub classes: HashMap<Ident, ClassInfo>,
    pub instances: Vec<InstanceInfo>,
    pub functions: HashMap<Ident, FuncInfo>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            adts: HashMap::new(),
            classes: HashMap::new(),
            instances: Vec::new(),
            functions: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdtInfo {
    pub name: Ident,
    pub type_vars: Vec<Ident>,
    pub constructors: Vec<Constructor>,
}

impl From<AdtDecl> for AdtInfo {
    fn from(decl: AdtDecl) -> Self {
        AdtInfo {
            name: decl.name,
            type_vars: decl.type_vars,
            constructors: decl.constructors,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClassInfo {
    pub name: Ident,
    pub type_vars: Vec<Ident>,
    pub signature: Option<TypeRef>,
}

#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub class_name: Ident,
    pub types: Vec<TypeRef>,
    pub constraints: Vec<Constraint>,
    pub members: Vec<FuncDecl>,
}

#[derive(Debug, Clone)]
pub struct FuncInfo {
    pub name: Ident,
    pub signature: Option<TypeRef>,
    pub args: Vec<Ident>,
    pub body: Expr,
    pub where_block: Option<Vec<FuncDecl>>,
}

impl From<FuncDecl> for FuncInfo {
    fn from(decl: FuncDecl) -> Self {
        FuncInfo {
            name: decl.name,
            signature: decl.signature,
            args: decl.args,
            body: decl.body,
            where_block: decl.where_block,
        }
    }
}
