
// Abstract Syntax Tree Definitions
// Based on docs/ast.md & docs/spec.md

pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Import(ImportDecl),
    Type(TypeDecl),
    Behavior(BehaviorDecl),
    Function(FuncDecl),
    Flow(FlowDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub path: String,
}

// 2. Type Definitions

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDecl {
    pub name: Ident,
    // Spec: type A : B C
    pub parent_constraint: Option<Constraint>, 
    // Spec: type A { prop, ... }
    pub properties: Vec<Ident>,
    pub components: Option<Vec<Ident>>,
    pub structure: Option<Ident>,
}

// 3. Logic Definitions (Behaviors & Functions)

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorDecl {
    pub name: Ident,
    // Spec: behavior Name(arg: Constraint, ...) -> Constraint
    pub args: Vec<TypedArg>,
    pub return_type: Constraint, 
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: Ident,
    pub params: Vec<TypedArg>, 
    pub return_type: Constraint,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedArg {
    pub name: Ident,
    pub constraint: Constraint,
}

// Spec: Constraints
// Addition: (A B)
// Union: (A | B)
// Subtraction: (A - B)
#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    Atom(Ident),           // "Series", "NonZero", "'a"
    Addition(Vec<Constraint>), // Intersection / Composition
    Union(Vec<Constraint>),    // Or
    Subtraction(Box<Constraint>, Box<Constraint>), // A - B
    None, // Empty constraint
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>, 
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Flow(FlowStmt),
    Return(Expr),
    Expr(Expr),
}

// 4. Flow Logic

#[derive(Debug, Clone, PartialEq)]
pub struct FlowDecl {
    pub name: Ident,
    pub body: Vec<FlowStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowStmt {
    // x = expr
    Assignment {
        target: Ident,
        expr: Expr,
    },
    Return(Expr),
}

// 5. Expressions

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    BinaryOp { left: Box<Expr>, op: Op, right: Box<Expr> },
    UnaryOp { op: Op, target: Box<Expr> },
    Call { path: Path, args: Vec<ArgValue> },
    MemberAccess { target: Box<Expr>, field: Ident },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add, Sub, Mul, Div, Eq, Neq, Lt, Gt, And, Or, Not
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub segments: Vec<Ident>, 
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArgValue {
    pub name: Option<Ident>,
    pub value: Expr,
}
