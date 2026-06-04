pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub enum FlowStmt {
    Assignment { target: Ident, expr: Expr },
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Call {
        fn_name: Ident,
        args: Vec<Expr>,
    },
    List(Vec<Expr>),
    Range {
        start: Box<Expr>,
        step: Option<Box<Expr>>,
        end: Box<Expr>,
    },
} // Question : do we allow expressions in list like [ multiply(1,2), 3 ]? should we allow or not?

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}

impl Into<(f64, f64, f64)> for &Literal {
    fn into(self) -> (f64, f64, f64) {
        match self {
            Literal::Integer(x) => (*x as f64, 0.0, 0.0),
            Literal::Float(x) => (0.0, *x, 0.0),
            Literal::String(_) => (0.0, 0.0, 1.0),
            Literal::Boolean(_) => (0.0, 0.0, 2.0),
        }
    }
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

use std::fmt;

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
            Expr::Call { fn_name, args } => {
                let formatted_args: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                write!(f, "{}({})", fn_name, formatted_args.join(", "))
            }
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
