// use stdlib::OperatorSpec;

// use crate::ast::{Network, Node, NodeType};
// use crate::expr::Literal;

// #[derive(Debug, Clone, PartialEq)]
// pub struct PolishExpr {
//     pub tokens: Vec<Token>,
//     pub arity: usize,
// }

// impl PolishExpr {
//     pub fn new() -> Self {
//         Self {
//             tokens: vec![],
//             arity: 0,
//         }
//     }
// }

// #[derive(Debug, Clone, PartialEq)]
// pub enum Token {
//     Operator(OperatorSpec),
//     Literal(Literal),
//     Parameter(usize),
//     Behavior(crate::behavior::BehaviorDecl),
// }

// impl Token {
//     pub fn pops(&self) -> usize {
//         match self {
//             Token::Operator(op) => op.inputs.len(),
//             Token::Literal(_) => 0,
//             Token::Parameter(_) => 0,
//             Token::Behavior(b) => b.inputs.len(),
//         }
//     }

//     pub fn pushes(&self) -> usize {
//         1
//     }
// }

// pub fn to_rpn(network: &Network, root: usize) -> PolishExpr {
//     let mut expr = Vec::new();
//     let node = &network.nodes[root];
//     for &child in &node.children {
//         expr.extend(to_rpn(network, child).tokens);
//     }
//     match &node.node_type {
//         NodeType::Literal(lit) => expr.push(Token::Literal(lit.clone())),
//         NodeType::Operator(op) => expr.push(Token::Operator(op.clone())),
//         NodeType::Behavior(b) => expr.push(Token::Behavior(b.clone())),
//     }
//     PolishExpr {
//         tokens: expr,
//         arity: 0,
//     }
// }

// pub fn from_rpn(expr: &PolishExpr) -> Network {
//     let mut network = Network::new();
//     if expr.tokens.is_empty() {
//         return network;
//     }

//     let mut iter = expr.tokens.iter().rev();

//     fn parse_recursive<'a>(
//         network: &mut Network,
//         iter: &mut impl Iterator<Item = &'a Token>,
//     ) -> usize {
//         let token = iter.next().expect("Invalid RPN");
//         match token {
//             Token::Literal(lit) => network.add_node(NodeType::Literal(lit.clone())),
//             Token::Operator(op) => {
//                 let idx = network.add_node(NodeType::Operator(op.clone()));
//                 let mut children = Vec::new();
//                 for _ in 0..op.inputs.len() {
//                     children.push(parse_recursive(network, iter));
//                 }
//                 children.reverse();
//                 for child in children {
//                     network.add_child(idx, child);
//                 }
//                 idx
//             }
//             Token::Behavior(b) => {
//                 let idx = network.add_node(NodeType::Behavior(b.clone()));
//                 let mut children = Vec::new();
//                 for _ in 0..b.inputs.len() {
//                     children.push(parse_recursive(network, iter));
//                 }
//                 children.reverse();
//                 for child in children {
//                     network.add_child(idx, child);
//                 }
//                 idx
//             }
//             Token::Parameter(_) => {
//                 panic!("Parameter token should not exist in final RPN");
//             }
//         }
//     }

//     parse_recursive(&mut network, &mut iter);
//     network
// }

// #[cfg(test)]
// mod tests {
//     use stdlib::types::Signal;

//     use crate::behavior::BehaviorDecl;

//     use super::*;
//     #[test]
//     fn test_to_rpn() {
//         // * Example flow:
//         // a = data("x")
//         // b = data("y")
//         // c = Mix(a, b)
//         // add(c, b)

//         // * A possible instance of behavior Mix :
//         // ShiftParam(0), ShiftParam(1), Reduce("Mix")

//         // * Initial call graph before Mix instantiation (index may be shuffled) :
//         // Equation : add(Mix(data("x"), data("y")), data("y"))
//         // 0 : Operator(add), params : [1,3]
//         // 1 : Behavior(Mix), params : [2,3]
//         // 2 : Operator(data), params : [4]
//         // 3 : Operator(data), params : [5]
//         // 4 : "x", params : []
//         // 5 : "y", params : []
//         // undetermined node index : 1 (Mix)
//         let add_spec = OperatorSpec::from("add");
//         let mix_spec = BehaviorDecl::new(
//             "mix",
//             vec![Signal::DataFrame(None), Signal::DataFrame(None)],
//             Signal::DataFrame(None),
//         );
//         let data_x_spec = OperatorSpec::from("data");
//         let data_y_spec = OperatorSpec::from("data");
//         let network = Network {
//             nodes: vec![
//                 Node {
//                     node_type: NodeType::Operator(add_spec),
//                     children: vec![1, 3],
//                 },
//                 Node {
//                     node_type: NodeType::Behavior(mix_spec),
//                     children: vec![2, 3],
//                 },
//                 Node {
//                     node_type: NodeType::Operator(data_x_spec),
//                     children: vec![4],
//                 },
//                 Node {
//                     node_type: NodeType::Operator(data_y_spec),
//                     children: vec![5],
//                 },
//                 Node {
//                     node_type: NodeType::Literal(Literal::String("x".to_string())),
//                     children: vec![],
//                 },
//                 Node {
//                     node_type: NodeType::Literal(Literal::String("y".to_string())),
//                     children: vec![],
//                 },
//             ],
//         };
//         println!("{}", network.format_node(0));
//         let rpn = to_rpn(&network, 0);
//         let network_recovered = from_rpn(&rpn);
//         println!("{}", network_recovered.format_node(0));
//         let rpn_recovered = to_rpn(&network_recovered, 0);
//         // assert_eq!(network, network_recovered);
//         assert_eq!(rpn, rpn_recovered);
//     }
// }

// // fn test_parse_behavior_decl() {
// //     let input = r#"

// // //         match arg {
// // //             Tree::Operator(p) => {
// // //                 if let Some(pe) = &p.polish_expression {
// // //                     polish_expr.extend(pe.clone());
// // //                 }
// // //             }
// // //             Tree::Literal(l) => polish_expr.push(Token::Literal(l.clone())),
// // //             Tree::Behavior(_) => panic!("Behavior cannot be an argument to Program"),
// // //         }
// // //     }
// // //     polish_expr.push(Token::Operator(spec.clone()));

// // impl Into<String> for &Token {
// //     fn into(self) -> String {
// //         match self {
// //             Token::Operator(operator) => format!("op!{}", operator.name),
// //             Token::Parameter(index) => format!("param!{}", index),
// //             Token::Literal(Literal::Boolean(b)) => format!("bool!{}", b),
// //             Token::Literal(Literal::Integer(x)) => format!("int!{}", x),
// //             Token::Literal(Literal::Float(x)) => format!("float!{}", x),
// //             Token::Literal(Literal::String(x)) => format!("str!{}", x),
// //         }
// //     }
// // }

// // // Polish expression in String, for cache key
// // impl Into<String> for &OperatorNode {
// //     fn into(self) -> String {
// //         let s: Vec<String> = self
// //             .polish_expression
// //             .as_ref()
// //             .unwrap()
// //             .iter()
// //             .map(|token| token.into())
// //             .collect();
// //         s.join(" ")
// //     }
// // }

// // // Convert single program to tree
// // impl From<OperatorNode> for Tree {
// //     fn from(param: OperatorNode) -> Self {
// //         let tokens = vec![Token::Parameter(0)];
// //         let params = vec![param];
// //         (&tokens, params).into()
// //     }
// // }

// // // From tokens (with parameter shift) and parameter list (Program), generate a AST.
// // impl From<(&PolishExpr, Vec<OperatorNode>)> for Tree {
// //     // (tokens, params)
// //     fn from((tokens, params): (&PolishExpr, Vec<OperatorNode>)) -> Self {
// //         let mut arglist: Vec<Tree> = Vec::new();
// //         for token in tokens.iter() {
// //             match token {
// //                 Token::Operator(operator) => {
// //                     let arity = operator.inputs.len();
// //                     if arglist.len() < arity {
// //                         panic!("Stack underflow for operator {}", operator.name);
// //                     }
// //                     let mut _arglist = arglist.split_off(arglist.len() - arity);
// //                     //_arglist.reverse();

// //                     let mut polish_expr: Vec<Token> = Vec::new();
// //                     for arg in _arglist.clone() {
// //                         match arg {
// //                             Tree::Operator(program) => {
// //                                 polish_expr.extend(program.polish_expression.unwrap())
// //                             }
// //                             Tree::Literal(literal) => polish_expr.push(Token::Literal(literal)),
// //                             Tree::Behavior(_) => panic!("Unexpected behavior in polish expression"),
// //                         }
// //                     }
// //                     polish_expr.push(Token::Operator(operator.clone()));
// //                     let program = OperatorNode {
// //                         spec: operator.clone(),
// //                         parameters: Some(_arglist),
// //                     };

// //                     arglist.push(Tree::Operator(program));
// //                 }
// //                 Token::Literal(literal) => {
// //                     arglist.push(Tree::Literal(literal.clone()));
// //                 }
// //                 Token::Parameter(index) => {
// //                     arglist.push(Tree::Operator(params[*index].clone()));
// //                 }
// //             }
// //         }
// //         if arglist.len() != 1 {
// //             panic!(
// //                 "Polish expression to ast conversion failed : \n Resulting trees : \n{:?}",
// //                 arglist
// //             );
// //         } else {
// //             arglist.pop().unwrap()
// //         }
// //     }
// // }
