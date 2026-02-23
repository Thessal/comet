
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

// (User-defined Type nodes are no longer used. See primitives.)

#[derive(Debug, Clone, PartialEq)]
pub enum TypeDecl {
    Series,
    DataFrame,
    Matrix,
    Vector,
    String,
    Int,
    Float,
    Bool,
}

// 3. Logic Definitions (Behaviors & Functions)

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorDecl {
    pub name: Ident,
    // Spec: behavior Name(arg: Constraint, ...) -> Constraint
    pub args: Vec<TypedArg>,
    pub return_constraint: ConstraintDecl, 
    pub depth: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: Ident,
    pub params: Vec<TypedArg>, 
    pub return_constraint: ConstraintDecl,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedArg {
    pub name: Ident,
    pub constraint: ConstraintDecl,
}

// Spec: Categories
// Addition: (A B)
// Union: (A | B)
// Subtraction: (A - B)
#[derive(Debug, Clone, PartialEq)]
pub enum CategoryExpr {
    Atom(Ident),           // "NonZero", "'a"
    Addition(Vec<CategoryExpr>), // Intersection / Composition
    Union(Vec<CategoryExpr>),    // Or
    Subtraction(Box<CategoryExpr>, Box<CategoryExpr>), // A - B
    None, // Empty category
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategorySetDecl {
    pub name: Option<Ident>, // e.g. 'a, 'b
    pub categories: Vec<CategoryExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintDecl {
    pub base_type: TypeDecl,
    pub category_expr: Option<CategoryExpr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Stmt>, 
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Flow(FlowStmt),
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
    Expr(Expr),
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

use std::fmt;

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for decl in &self.declarations {
            writeln!(f, "{}", decl)?;
        }
        Ok(())
    }
}

impl fmt::Display for Declaration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Declaration::Import(i) => write!(f, "{}", i),
            Declaration::Type(t) => write!(f, "{}", t),
            Declaration::Behavior(b) => write!(f, "{}", b),
            Declaration::Function(fun) => write!(f, "{}", fun),
            Declaration::Flow(flow) => write!(f, "{}", flow),
        }
    }
}

impl fmt::Display for ImportDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "import {:?}", self.path)
    }
}

impl fmt::Display for TypeDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeDecl::Series => write!(f, "Series"),
            TypeDecl::DataFrame => write!(f, "DataFrame"),
            TypeDecl::Matrix => write!(f, "Matrix"),
            TypeDecl::Vector => write!(f, "Vector"),
            TypeDecl::String => write!(f, "String"),
            TypeDecl::Int => write!(f, "Int"),
            TypeDecl::Float => write!(f, "Float"),
            TypeDecl::Bool => write!(f, "Bool"),
        }
    }
}

impl fmt::Display for CategoryExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CategoryExpr::Atom(name) => write!(f, "{}", name),
            CategoryExpr::Addition(cats) => {
                let s: Vec<String> = cats.iter().map(|c| c.to_string()).collect();
                write!(f, "{}", s.join(" "))
            },
            CategoryExpr::Union(cats) => {
                let s: Vec<String> = cats.iter().map(|c| c.to_string()).collect();
                write!(f, "({})", s.join(" | "))
            },
            CategoryExpr::Subtraction(lhs, rhs) => {
                write!(f, "{} - {}", lhs, rhs)
            },
            CategoryExpr::None => write!(f, ""),
        }
    }
}

impl fmt::Display for ConstraintDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base_type)?;
        if let Some(cat) = &self.category_expr {
            write!(f, " {}", cat)?;
        }
        Ok(())
    }
}

impl fmt::Display for CategorySetDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(n) = &self.name {
            write!(f, "{}: ", n)?;
        }
        let cats: Vec<String> = self.categories.iter().map(|c| c.to_string()).collect();
        write!(f, "{}", cats.join(", "))
    }
}

impl fmt::Display for TypedArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.constraint)
    }
}

impl fmt::Display for BehaviorDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| a.to_string()).collect();
        write!(f, "Behavior {}({}) -> {}\n", self.name, args.join(", "), self.return_constraint)
    }
}

impl fmt::Display for FuncDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let params: Vec<String> = self.params.iter().map(|a| a.to_string()).collect();
        write!(f, "Fn {}({}) -> {}\n", self.name, params.join(", "), self.return_constraint)
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{{")?;
        for stmt in &self.stmts {
            writeln!(f, "    {}", stmt)?;
        }
        writeln!(f, "}}")
    }
}

impl fmt::Display for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::Flow(fs) => write!(f, "{}", fs),
            Stmt::Expr(e) => write!(f, "{}", e), // Return
        }
    }
}

impl fmt::Display for FlowDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Flow {} {{", self.name)?;
        for stmt in &self.body {
            writeln!(f, "    {}", stmt)?;
        }
        writeln!(f, "}}")
    }
}

impl fmt::Display for FlowStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowStmt::Assignment { target, expr } => write!(f, "{} = {}", target, expr),
            FlowStmt::Expr(e) => write!(f, "{}", e),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Op::Add => "+", Op::Sub => "-", Op::Mul => "*", Op::Div => "/",
            Op::Eq => "==", Op::Neq => "!=", Op::Lt => "<", Op::Gt => ">",
            Op::And => "&&", Op::Or => "||", Op::Not => "!",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(l) => write!(f, "{}", l),
            Expr::Identifier(i) => write!(f, "{}", i),
            Expr::BinaryOp { left, op, right } => write!(f, "{} {} {}", left, op, right),
            Expr::UnaryOp { op, target } => write!(f, "{}{}", op, target),
            Expr::Call { path, args } => {
                let a: Vec<String> = args.iter().map(|a| a.to_string()).collect();
                write!(f, "{}({})", path, a.join(", "))
            },
            Expr::MemberAccess { target, field } => write!(f, "{}.{}", target, field),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Integer(i) => write!(f, "{}", i),
            Literal::Float(fl) => write!(f, "{}", fl),
            Literal::String(s) => write!(f, "\"{}\"", s),
            Literal::Boolean(b) => write!(f, "{}", b),
        }
    }
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.segments.join("::"))
    }
}

impl fmt::Display for ArgValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(n) = &self.name {
            write!(f, "{}={}", n, self.value)
        } else {
            write!(f, "{}", self.value)
        }
    }
}
