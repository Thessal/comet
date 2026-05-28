// #[derive(Clone, PartialEq)]
// pub enum Token {
//     Operator(OperatorSpec),
//     Literal(Literal),
//     Parameter(usize),
// }
// pub type PolishExpr = Vec<Token>;

// impl fmt::Debug for Token {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         match self {
//             Token::Operator(operator) => write!(f, "op!{}", operator.name),
//             Token::Literal(literal) => write!(f, "{}", literal),
//             Token::Parameter(index) => write!(f, "param!{}", index),
//         }
//     }
// }
// /// Helper to cleanly construct a Operator (and its polish expression) from an operator name and arguments

// // pub polish_expression: Option<PolishExpr>,

// //     let mut polish_expr = Vec::new();
// //     for arg in &args {
// //         match arg {
// //             Tree::Operator(p) => {
// //                 if let Some(pe) = &p.polish_expression {
// //                     polish_expr.extend(pe.clone());
// //                 }
// //             }
// //             Tree::Literal(l) => polish_expr.push(Token::Literal(l.clone())),
// //             Tree::Behavior(_) => panic!("Behavior cannot be an argument to Program"),
// //         }
// //     }
// //     polish_expr.push(Token::Operator(spec.clone()));

// impl Into<String> for &Token {
//     fn into(self) -> String {
//         match self {
//             Token::Operator(operator) => format!("op!{}", operator.name),
//             Token::Parameter(index) => format!("param!{}", index),
//             Token::Literal(Literal::Boolean(b)) => format!("bool!{}", b),
//             Token::Literal(Literal::Integer(x)) => format!("int!{}", x),
//             Token::Literal(Literal::Float(x)) => format!("float!{}", x),
//             Token::Literal(Literal::String(x)) => format!("str!{}", x),
//         }
//     }
// }

// // Polish expression in String, for cache key
// impl Into<String> for &OperatorNode {
//     fn into(self) -> String {
//         let s: Vec<String> = self
//             .polish_expression
//             .as_ref()
//             .unwrap()
//             .iter()
//             .map(|token| token.into())
//             .collect();
//         s.join(" ")
//     }
// }

// // Convert single program to tree
// impl From<OperatorNode> for Tree {
//     fn from(param: OperatorNode) -> Self {
//         let tokens = vec![Token::Parameter(0)];
//         let params = vec![param];
//         (&tokens, params).into()
//     }
// }

// // From tokens (with parameter shift) and parameter list (Program), generate a AST.
// impl From<(&PolishExpr, Vec<OperatorNode>)> for Tree {
//     // (tokens, params)
//     fn from((tokens, params): (&PolishExpr, Vec<OperatorNode>)) -> Self {
//         let mut arglist: Vec<Tree> = Vec::new();
//         for token in tokens.iter() {
//             match token {
//                 Token::Operator(operator) => {
//                     let arity = operator.inputs.len();
//                     if arglist.len() < arity {
//                         panic!("Stack underflow for operator {}", operator.name);
//                     }
//                     let mut _arglist = arglist.split_off(arglist.len() - arity);
//                     //_arglist.reverse();

//                     let mut polish_expr: Vec<Token> = Vec::new();
//                     for arg in _arglist.clone() {
//                         match arg {
//                             Tree::Operator(program) => {
//                                 polish_expr.extend(program.polish_expression.unwrap())
//                             }
//                             Tree::Literal(literal) => polish_expr.push(Token::Literal(literal)),
//                             Tree::Behavior(_) => panic!("Unexpected behavior in polish expression"),
//                         }
//                     }
//                     polish_expr.push(Token::Operator(operator.clone()));
//                     let program = OperatorNode {
//                         spec: operator.clone(),
//                         parameters: Some(_arglist),
//                     };

//                     arglist.push(Tree::Operator(program));
//                 }
//                 Token::Literal(literal) => {
//                     arglist.push(Tree::Literal(literal.clone()));
//                 }
//                 Token::Parameter(index) => {
//                     arglist.push(Tree::Operator(params[*index].clone()));
//                 }
//             }
//         }
//         if arglist.len() != 1 {
//             panic!(
//                 "Polish expression to ast conversion failed : \n Resulting trees : \n{:?}",
//                 arglist
//             );
//         } else {
//             arglist.pop().unwrap()
//         }
//     }
// }
