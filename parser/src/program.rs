// Program Definitions
// Based on docs/ast.md & docs/spec.md

pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Import(ImportDecl),
    Behavior(BehaviorDecl),

    Flow(FlowDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    pub path: String,
}

// 2. Type Definitions

// (User-defined Type nodes are no longer used. See primitives.)

pub const TYPE_DECL_LENGTH: usize = 7;
#[derive(Debug, Clone, PartialEq)]
pub enum TypeDecl {
    DataFrame,
    Matrix,
    Vector,
    String,
    Float,
    Bool,
    Void, // for statements that don't return values
}

// 3. Logic Definitions (Behaviors & Functions)

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<TypedArg>,
    pub return_type: TypeDecl,
    pub weights: Option<String>,
    pub train: Option<bool>,
    pub supervised_epochs: Option<usize>,
    pub operators: Option<Vec<Ident>>,
    pub integers: Option<Vec<i64>>,
    pub floats: Option<Vec<f64>>,
    pub strings: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedArg {
    pub name: Ident,
    pub type_decl: TypeDecl,
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
    Assignment { target: Ident, expr: Expr },
    Expr(Expr),
}

// 5. Expressions

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Call {
        path: Path,
        args: Vec<Expr>, // Function arguments (positional only)
    },
    MemberAccess {
        target: Box<Expr>,
        field: Ident,
    },
    List(Vec<Expr>),
    Range {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        end: Box<Expr>,
    },
} // Question : do we allow expressions in list like [ multiply(1,2), 3 ]? should we allow or not?

// #[derive(Debug, Clone, PartialEq)]
// pub enum Op {
//     Add,
//     Sub,
//     Mul,
//     Div,
//     Eq,
//     Neq,
//     Lt,
//     Gt,
//     And,
//     Or,
//     Not,
// }

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

impl Eq for Literal {}

use std::hash::{Hash, Hasher};
impl Hash for Literal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Literal::Integer(i) => {
                0_u8.hash(state);
                i.hash(state);
            }
            Literal::Float(f) => {
                1_u8.hash(state);
                f.to_bits().hash(state);
            }
            Literal::String(s) => {
                2_u8.hash(state);
                s.hash(state);
            }
            Literal::Boolean(b) => {
                3_u8.hash(state);
                b.hash(state);
            }
        }
    }
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
            Declaration::Behavior(b) => write!(f, "{}", b),

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
            TypeDecl::DataFrame => write!(f, "DataFrame"),
            TypeDecl::Matrix => write!(f, "Matrix"),
            TypeDecl::Vector => write!(f, "Vector"),
            TypeDecl::String => write!(f, "String"),
            TypeDecl::Float => write!(f, "Float"),
            TypeDecl::Bool => write!(f, "Bool"),
            TypeDecl::Void => write!(f, "Void"),
        }
    }
}

impl fmt::Display for TypedArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.type_decl)
    }
}

impl fmt::Display for BehaviorDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self.args.iter().map(|a| a.to_string()).collect();
        let mut props = Vec::new();
        if let Some(w) = &self.weights {
            props.push(format!("weights = \"{}\"", w));
        }
        if let Some(t) = self.train {
            props.push(format!("train = {}", t));
        }
        if let Some(ss) = self.supervised_epochs {
            props.push(format!("supervised_epochs = {}", ss));
        }
        if let Some(ops) = &self.operators {
            props.push(format!("operators = [{}]", ops.join(", ")));
        }
        if let Some(ints) = &self.integers {
            let s: Vec<String> = ints.iter().map(|i| i.to_string()).collect();
            props.push(format!("integers = [{}]", s.join(", ")));
        }
        if let Some(flts) = &self.floats {
            let s: Vec<String> = flts.iter().map(|f| f.to_string()).collect();
            props.push(format!("floats = [{}]", s.join(", ")));
        }
        if let Some(strs) = &self.strings {
            let s: Vec<String> = strs.iter().map(|s| format!("\"{}\"", s)).collect();
            props.push(format!("strings = [{}]", s.join(", ")));
        }

        if props.is_empty() {
            writeln!(
                f,
                "Behavior {}({}) -> {}",
                self.name,
                args.join(", "),
                self.return_type
            )
        } else {
            writeln!(
                f,
                "Behavior {}({}) {{ {} }} -> {}",
                self.name,
                args.join(", "),
                props.join(", "),
                self.return_type
            )
        }
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

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Literal(l) => write!(f, "{}", l),
            Expr::Identifier(i) => write!(f, "{}", i),
            Expr::Call { path, args } => {
                let formatted_args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", path, formatted_args.join(", "))
            }
            Expr::MemberAccess { target, field } => write!(f, "{}.{}", target, field),
            Expr::List(exprs) => {
                let s: Vec<String> = exprs.iter().map(|e| e.to_string()).collect();
                write!(f, "[{}]", s.join(", "))
            }
            Expr::Range { start, step, end } => {
                if let Some(st) = step {
                    write!(f, "[{}..{}..{}]", start, st, end)
                } else {
                    write!(f, "[{}..{}]", start, end)
                }
            }
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
