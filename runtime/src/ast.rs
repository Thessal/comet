use std::fmt;

use parser::expr::Literal;
use stdlib::OperatorSpec;
use stdlib::types::Signal;

//////////////
/* AST Node */
//////////////

#[derive(Clone, PartialEq)]
pub enum Token {
    Operator(OperatorSpec),
    Literal(Literal),
    Parameter(usize),
}

impl fmt::Debug for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Operator(operator) => write!(f, "op!{}", operator.name),
            Token::Literal(literal) => write!(f, "{}", literal),
            Token::Parameter(index) => write!(f, "param!{}", index),
        }
    }
}

impl Into<tch::Tensor> for &Token {
    fn into(self) -> tch::Tensor {
        let mut output: Vec<f64> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        match self {
            Token::Operator(_) => output[0] = 1.0,
            Token::Literal(x) => (output[1], output[2], output[3]) = x.into(),
            Token::Parameter(x) => output[4] = *x as f64,
        }
        tch::Tensor::from_slice(&output)
    }
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

impl Program {
    /// Helper to cleanly construct a Program (and its polish expression) from an operator name and arguments
    pub fn new(operator_name: &str, args: Vec<Tree>) -> Self {
        let spec = OperatorSpec::from(operator_name);
        
        let mut polish_expr = Vec::new();
        for arg in &args {
            match arg {
                Tree::Program(p) => {
                    if let Some(pe) = &p.polish_expression {
                        polish_expr.extend(pe.clone());
                    }
                }
                Tree::Literal(l) => polish_expr.push(Token::Literal(l.clone())),
            }
        }
        polish_expr.push(Token::Operator(spec.clone()));

        Program {
            spec,
            polish_expression: Some(polish_expr),
            parameters: Some(args),
        }
    }
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
            Token::Parameter(index) => format!("param!{}", index),
            Token::Literal(Literal::Boolean(b)) => format!("bool!{}", b),
            Token::Literal(Literal::Integer(x)) => format!("int!{}", x),
            Token::Literal(Literal::Float(x)) => format!("float!{}", x),
            Token::Literal(Literal::String(x)) => format!("str!{}", x),
        }
    }
}

// Convert single program to tree
impl From<Program> for Tree {
    fn from(param: Program) -> Self {
        let tokens = vec![Token::Parameter(0)];
        let params = vec![param];
        (&tokens, params).into()
    }
}

// From tokens (with parameter shift) and parameter list (Program), generate a AST.
impl From<(&PolishExpr, Vec<Program>)> for Tree {
    // (tokens, params)
    fn from((tokens, params): (&PolishExpr, Vec<Program>)) -> Self {
        let mut arglist: Vec<Tree> = Vec::new();
        for token in tokens.iter() {
            match token {
                Token::Operator(operator) => {
                    let arity = operator.inputs.len();
                    if arglist.len() < arity {
                        panic!("Stack underflow for operator {}", operator.name);
                    }
                    let mut _arglist = arglist.split_off(arglist.len() - arity);
                    //_arglist.reverse();

                    let mut polish_expr: Vec<Token> = Vec::new();
                    for arg in _arglist.clone() {
                        match arg {
                            Tree::Program(program) => {
                                polish_expr.extend(program.polish_expression.unwrap())
                            }
                            Tree::Literal(literal) => polish_expr.push(Token::Literal(literal)),
                        }
                    }
                    polish_expr.push(Token::Operator(operator.clone()));
                    let program = Program {
                        spec: operator.clone(),
                        polish_expression: Some(polish_expr),
                        parameters: Some(_arglist),
                    };

                    arglist.push(Tree::Program(program));
                }
                Token::Literal(literal) => {
                    arglist.push(Tree::Literal(literal.clone()));
                }
                Token::Parameter(index) => {
                    arglist.push(Tree::Program(params[*index].clone()));
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
    //     let op = stdlib::OperatorSpec::from(sig.clone());
    //     assert!(op.name == "ts_mean");
    //     assert!(op.inputs == vec![stdlib::TypeDecl::ScalarInt, stdlib::TypeDecl::DataFrame]);
    //     assert!(op.output_shape == stdlib::TypeDecl::DataFrame);
    // }
    // // TODO: from(sig) -> Op using wrong inputs/outpus.
}
