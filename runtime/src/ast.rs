use crate::runtime::Runtime;
use parser::expr::Literal;
use stdlib::types::Signal;

////////////////////////////////
/* stdlib wrapper for runtime */
////////////////////////////////

#[derive(Debug, Clone, PartialEq)]
pub struct OperatorSpec {
    pub name: String,
    pub inputs: Vec<Signal>,
    pub output: Signal,
}

impl OperatorSpec {
    fn fill(self, arglist: Vec<Tree>) -> Program {
        let arity = self.inputs.len();
        if arglist.len() < arity {
            panic!("Stack underflow for operator {}", self.name);
        }
        let mut args = Vec::with_capacity(arity);
        for arg in arglist {
            args.push(arg);
        }
        args.reverse();
        let mut polish_expr: Vec<Token> = Vec::new();
        for arg in args.clone() {
            match arg {
                Tree::Program(program) => polish_expr.extend(program.polish_expression.unwrap()),
                Tree::Literal(literal) => polish_expr.push(Token::Literal(literal)),
            }
        }
        polish_expr.push(Token::Operator(self.clone()));
        Program {
            spec: self,
            polish_expression: Some(polish_expr),
            parameters: Some(args),
        }
    }
}

impl From<&str> for OperatorSpec {
    fn from(name: &str) -> Self {
        let op = OperatorMeta::from(name);
        OperatorSpec {
            name: op.name.to_string(),
            inputs: op.inputs.to_vec(),
            output: op.output_shape,
        }
    }
}

use stdlib::OperatorMeta;
//////////////
/* AST Node */
//////////////

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Operator(OperatorSpec),
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Tree {
    Program(Program),
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub spec: OperatorSpec,
    pub polish_expression: Option<PolishExpr>,
    pub parameters: Option<Vec<Tree>>,
}
pub type PolishExpr = Vec<Token>;

// Polish expression in String, for cache key
impl Into<String> for &Program {
    fn into(self) -> String {
        let s: Vec<String> = self
            .polish_expression
            .as_ref()
            .unwrap()
            .iter()
            .map(|token| token.into())
            .collect();
        s.join(" ")
    }
}

impl Into<String> for &Token {
    fn into(self) -> String {
        match self {
            Token::Operator(operator) => format!("op!{}", operator.name),
            Token::Literal(Literal::Boolean(b)) => format!("bool!{}", b),
            Token::Literal(Literal::Integer(x)) => format!("int!{}", x),
            Token::Literal(Literal::Float(x)) => format!("float!{}", x),
            Token::Literal(Literal::String(x)) => format!("str!{}", x),
        }
    }
}

impl From<&PolishExpr> for Tree {
    fn from(tokens: &Vec<Token>) -> Self {
        let mut arglist: Vec<Tree> = Vec::new();
        for token in tokens.iter() {
            match token {
                Token::Operator(operator) => {
                    let arity = operator.inputs.len();
                    if arglist.len() < arity {
                        panic!("Stack underflow for operator {}", operator.name);
                    }
                    let mut _arglist = arglist.split_off(arglist.len() - arity);
                    _arglist.reverse();
                    let program = operator.clone().fill(_arglist);
                    arglist.push(Tree::Program(program));
                }
                Token::Literal(literal) => {
                    arglist.push(Tree::Literal(literal.clone()));
                }
            }
        }
        if arglist.len() != 1 {
            panic!(
                "Polish expression to ast conversion failed : \n Resulting trees : \n{:?}",
                arglist
            );
        } else {
            arglist.pop().unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    // #[test]
    // fn test() {
    //     let sig = crate::ast::OperatorSpec {
    //         name: "ts_mean".into(),
    //         inputs: vec![
    //             parser::program::TypeDecl::Integer,
    //             parser::program::TypeDecl::DataFrame,
    //         ],
    //         output: parser::program::TypeDecl::DataFrame,
    //     };
    //     let op = stdlib::OperatorMeta::from(sig.clone());
    //     assert!(op.name == "ts_mean");
    //     assert!(op.inputs == vec![stdlib::TypeDecl::ScalarInt, stdlib::TypeDecl::DataFrame]);
    //     assert!(op.output_shape == stdlib::TypeDecl::DataFrame);
    // }
    // // TODO: from(sig) -> Op using wrong inputs/outpus.
}
