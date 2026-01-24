
// Abstract Syntax Tree Definitions
// Based on docs/ast.md

pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Import(ImportDecl),
    Type(TypeDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Behavior(BehaviorDecl),
    Impl(ImplDecl),
    Function(FuncDecl),
    Flow(FlowDecl),
    Property(PropertyDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub path: String,
}

// 2. Type Definitions

#[derive(Debug, Clone, PartialEq)]
pub struct TypeDecl {
    pub name: Ident,
    pub parent: Ident,
    pub properties: Vec<Ident>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: Ident,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumDecl {
    pub name: Ident,
    pub variants: Vec<Ident>, // Simplified for now, docs just said EnumVariantList
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: Ident,
    pub ty: TypeRef,
}

pub type TypeRef = String; // Placeholder for now, docs used TypeRef but didn't define it fully in the snippet.

// 3. Logic Definitions (Behaviors & Impls)

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<Ident>,
    pub return_type: Option<Ident>, // -> Identifier is optional in some languages, but here it looks required? Docs: "-> Identifier"
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplDecl {
    pub name: Ident, // Unique name (e.g., "Ratio")
    pub behavior: Ident,
    pub args: Vec<Ident>, // e.g., ["A", "B"]
    pub constraints: Option<Expr>, // "where B is NonZero"
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: Ident,
    pub params: Vec<Param>, // ParamList
    pub return_type: Ident,
    pub constraints: Option<Vec<Constraint>>, // ConstraintList?
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: TypeRef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>, // Block not fully defined in ast.md snippet, assuming Stmt list
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Flow(FlowStmt),
    // ... other stmts
}


#[derive(Debug, Clone, PartialEq)]
pub struct PropertyDecl {
    pub name: Ident,
}

// 4. Flow Logic

#[derive(Debug, Clone, PartialEq)]
pub struct FlowDecl {
    pub name: Ident,
    pub body: Vec<FlowStmt>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FlowStmt {
    Generator {
        target: Ident,
        source: Expr, // e.g. "Universe(Earnings)" or "Comparator(x, y)"
        constraints: Option<Expr>,
    },
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
    Call { path: Path, args: Vec<Expr> },
    MemberAccess { target: Box<Expr>, field: Ident },
    PropertyCheck { target: Box<Expr>, property: Ident }, // "is NonZero"
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add, Sub, Mul, Div, Eq, Neq, Lt, Gt, And, Or, Not
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub segments: Vec<Ident>, // e.g., ["Comparator", "compare"]
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}
