use crate::action::{Action, ActionSpace};

use parser::ast::{Network, Node, NodeType};
use parser::behavior::BehaviorDecl;
use parser::expr::Literal;
use parser::polish::{PolishExpr, Token};

use runtime::runtime::Runtime;
use stdlib::OperatorSpec;
use stdlib::types::Signal;

//////////
/// Search State
//////////

#[derive(Debug, Clone)]
struct AbstractMachine {
    // A signal is represented as a root node in callgraph, and an item in stack.
    // 1. When you introduce a function parameter, you add a node in callgraph,
    // and add a new item in stack.
    // 2. When you reduce an operator, you merge root nodes in callgraph into a single node,
    // and pop the arguments from stack. Push the output signal to stack.
    stack: Vec<Signal>, // instructions
    addr: Vec<usize>,   // address of root nodes
    callgraph: Network, // memory
}

impl AbstractMachine {
    fn check_reduce(&self, operator_spec: &OperatorSpec) -> bool {
        // type checking
        assert!(operator_spec.inputs.len() <= self.stack.len());
        for (op_input, stack_item) in operator_spec.inputs.iter().zip(self.stack.iter().rev()) {
            if std::mem::discriminant(op_input) != std::mem::discriminant(stack_item) {
                return false;
            }
        }
        true
    }

    // TODO: addr and callgraph update



    fn update_ast(&mut self, action: &Action) {

        match action {
                Action::ShiftInt(i) => {
                    let idx = self
                        .callgraph
                        .add_node(NodeType::Literal(Literal::Integer(*i)));
                }
                Action::ShiftFloat(f) => {
                    let idx = self
                        .callgraph
                        .add_node(NodeType::Literal(Literal::Float(*f)));
                }
                Action::ShiftString(s) => {
                    let idx = self
                        .callgraph
                        .add_node(NodeType::Literal(Literal::String(s.clone())));
                }
                Action::ShiftParam(p) => {
                    node_stack.push(behavior_children[*p]);
                }
                Action::Reduce(op) => {
                    let idx = self.callgraph.add_node(NodeType::Operator(op.clone()));
                    let arity = op.inputs.len();
                    let mut children = Vec::new();
                    for _ in 0..arity {
                        children.push(node_stack.pop().expect("Stack underflow in Reduce"));
                    }
                    children.reverse();
                    for child in children {
                        self.callgraph.add_child(idx, child);
                    }
                    node_stack.push(idx);
                }
                Action::Done => unreachable!(),
            }
        }

        // let mut node_stack: Vec<usize> = Vec::new();
        // let behavior_children = self.callgraph.nodes[self.cursor].children.clone();

        // if let Some(root) = node_stack.pop() {
        //     let root_node = self.callgraph.nodes[root].clone();
        //     self.callgraph.nodes[self.cursor] = root_node;
        // }
    }

    
    fn push(&mut self, instruction: &Action) {
        match instruction {
            Action::Reduce(operator_spec) => {
                assert!(self.check_reduce(operator_spec));
                // pop arguments
                self.stack
                    .truncate(self.stack.len() - operator_spec.inputs.len());
            }
            Action::Done => unreachable!(),
            _ => {}
        }
        let signal = self.action_to_signal(instruction);
        self.stack.push(signal);
    }

    fn action_to_signal(&self, action: &Action) -> Signal {
        match action {
            Action::ShiftInt(i) => Signal::Int(Some(*i)),
            Action::ShiftFloat(f) => Signal::Float(Some(*f)),
            Action::ShiftString(s) => Signal::String(Some(s.clone())),
            Action::ShiftParam(i) => self.params[*i].clone(),
            Action::Reduce(op_spec) => op_spec.output_shape.clone(),
            Action::Done => unreachable!(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchState {
    pub machine: AbstractMachine, // used to track arguments and actions of episode
    pub callgraph: Network,       // full callgraph of the flow, including outside the behavior.
    pub cursor: usize, // index of node we are currently building. (location of expr in callgraph)
}

impl From<Network> for SearchState {
    fn from(callgraph: Network) -> Self {
        let (behavior_idx, behavior_decl) = callgraph.get_behavior();
        Self {
            machine: AbstractMachine {
                stack: vec![],
                params: behavior_decl.inputs.clone(),
            },
            callgraph,
            cursor: behavior_idx,
        }
    }
}

impl SearchState {
    pub fn new_dummy() -> Self {
        Self {
            machine: AbstractMachine {
                stack: vec![],
                params: vec![],
            },
            callgraph: Network { nodes: vec![] },
            cursor: 0,
        }
    }
    pub fn is_done_valid(&self) -> bool {
        // There may be unused inputs. We allow unused inputs.
        self.machine.stack.len() == 1
    }

    pub fn apply_action(&mut self, action: &Action) {
        match action {
            Action::Done => {
                // TODO: Update callgraph
                // Replace the current behavior node with the new one.
            }
            a => {
                self.machine.push(a);
                self.update_ast(a);
            }
        }
    }

    // pub fn get_valid_actions(&self, action: &ActionSpace) -> Vec<Action> {
    //     let stack_type_and_data: Vec<Signal> =
    //         self.stack.iter().map(|(_, _, s)| s.clone()).collect();
    //     let mut valid_actions: Vec<Action> = vec![];
    //     for i in 0..action.size() {
    //         let a = action.get_action(i);
    //         let is_valid: bool = match a {
    //             Action::Reduce(OperatorSpec {
    //                 name: _name,
    //                 inputs,
    //                 output_shape: _output_shape,
    //                 execute: _,
    //             }) => {
    //                 if inputs.len() > self.stack.len() {
    //                     false
    //                 } else {
    //                     inputs
    //                         .iter()
    //                         .rev()
    //                         .zip(stack_type_and_data.iter().rev())
    //                         // compare variant only. (not data)
    //                         .all(|(i, s)| std::mem::discriminant(i) == std::mem::discriminant(s))
    //                 }
    //             }
    //             Action::Done => self.is_done_valid(),
    //             _ => self.stack.len() < 5, // allow introducing variables until stack depth 5
    //                                        // TODO: adjust
    //         };
    //         if is_valid {
    //             valid_actions.push(action.get_action(i));
    //         }
    //     }
    //     valid_actions
    // }

    // fn make_tree(
    //     &self,
    //     action: Action,
    //     param_values: &Vec<usize>,
    //     network: &mut Network,
    //     runtime: &mut Runtime,
    // ) -> (SearchState, PolishExpr, usize, Signal) {
    //     let mut next_state = self.clone();
    //     let (expr, tree, data): (PolishExpr, usize, Signal) = match action {
    //         Action::ShiftInt(i) => (
    //             vec![Token::Literal(Literal::Integer(i))],
    //             network.add_node(NodeType::Literal(Literal::Integer(i))),
    //             Signal::Int(Some(i)),
    //         ),
    //         Action::ShiftFloat(f) => (
    //             vec![Token::Literal(Literal::Float(f))],
    //             network.add_node(NodeType::Literal(Literal::Float(f))),
    //             Signal::Float(Some(f)),
    //         ),
    //         Action::ShiftString(s) => (
    //             vec![Token::Literal(Literal::String(s.clone()))],
    //             network.add_node(NodeType::Literal(Literal::String(s.clone()))),
    //             Signal::String(Some(s.clone())),
    //         ),
    //         Action::ShiftParam(i) => {
    //             let (_param_name, _output_type, _used) = next_state.params[i].clone();
    //             next_state.params[i].2 = true;
    //             let tree = param_values[i];
    //             (vec![Token::Parameter(i)], tree, runtime.run(network, tree))
    //         }
    //         Action::Reduce(op) => {
    //             let arity = op.inputs.len();
    //             let inputs = next_state.stack.split_off(next_state.stack.len() - arity);
    //             let exprs: Vec<PolishExpr> = inputs.iter().map(|x| x.0.clone()).collect();
    //             let trees: Vec<usize> = inputs.iter().map(|x| x.1.clone()).collect();
    //             let datas: Vec<Signal> = inputs.iter().map(|x| x.2.clone()).collect();

    //             assert!(datas.len() == arity);
    //             // operator arguments type check - RPN
    //             assert!(
    //                 datas
    //                     .iter()
    //                     .zip(op.inputs.iter())
    //                     .all(|(d, i)| { std::mem::discriminant(d) == std::mem::discriminant(i) })
    //             );

    //             let expr: PolishExpr = exprs
    //                 .into_iter()
    //                 .chain(vec![vec![Token::Operator(op.clone())]].into_iter())
    //                 .flatten()
    //                 .collect();

    //             let tree = network.add_node(NodeType::Operator(op.clone()));
    //             for child in trees {
    //                 network.add_child(tree, child);
    //             }

    //             let data = runtime.run(network, tree);
    //             (expr, tree, data)
    //         }
    //         Action::Done => panic!("Done action should be handled in prior step."),
    //     };
    //     (next_state, expr, tree, data)
    // }
}

// #[cfg(test)]
// mod tests {
//     use runtime::runtime::test_make_param0;

//     use super::*;
//     #[test]
//     fn test_get_valid_actions() {
//         let mut runtime = Runtime::new(100, "../data".into(), None);
//         let behavior = BehaviorDecl {
//             inputs: vec![("x".to_string(), Signal::DataFrame(None))],
//             output: ("result".to_string(), Signal::DataFrame(None)),
//             operators: Some(vec!["ts_mean".to_string()]),
//             integers: Some(vec![0, 1]),
//             floats: Some(vec![0.0, 3.0]),
//             strings: None,
//             weights: None,
//             train: None,
//             supervised_epochs: None,
//         };
//         let action_space: ActionSpace = (&behavior).into();
//         let state = SearchState::from(behavior);
//         let valid_actions = state.get_valid_actions(&action_space);
//         assert_eq!(valid_actions.len(), 5); //[ShiftInt(0), ShiftInt(1), ShiftFloat(0.0), ShiftFloat(3.0), ShiftParam(0)]
//         assert!(valid_actions.contains(&Action::ShiftInt(0)));
//         assert!(!valid_actions.contains(&Action::ShiftInt(-1)));
//         assert!(valid_actions.contains(&Action::ShiftFloat(0.0)));
//         assert!(!valid_actions.contains(&Action::ShiftFloat(-1.0)));

//         let param_values: Vec<Tree> = vec![Tree::Program(test_make_param0())];
//         let (state, _1) = state.apply_action(Action::ShiftParam(0), &mut runtime, &param_values);
//         let (state, _1) = state.apply_action(Action::ShiftInt(0), &mut runtime, &param_values);
//         // println!("{:?}", state.stack);
//         // println!("action_space {:?}", action_space);
//         let valid_actions2 = state.get_valid_actions(&action_space);
//         // println!("valid_actions2 {:?}", valid_actions2);
//         assert!(valid_actions2.contains(&Action::Reduce(OperatorSpec::from("ts_mean"))));
//         assert_eq!(valid_actions2.len(), 6); // [ShiftInt(0), ShiftInt(1), ShiftFloat(0.0), ShiftFloat(3.0), Reduce("ts_mean"), ShiftParam(0)]

//         let operator_offset = 1 + 2 + 2 + 0; // Action space layout [done, integers, floats, strings, operators, params]
//         let action = action_space.get_action(operator_offset);
//         assert!(matches!(action, Action::Reduce(_)));
//         let (state, _1) = state.apply_action(action, &mut runtime, &param_values);
//         let valid_actions3 = state.get_valid_actions(&action_space);
//         // println!(
//         //     "is_done_valid: is_done_valid, params_consumed, stack_size = {:?}",
//         //     state.is_done_valid()
//         // );
//         // println!("stack_size: {:?}", state.stack.len());
//         // println!("valid_actions3 {:?}", valid_actions3);
//         assert!(valid_actions3.contains(&Action::Done));
//         assert_eq!(valid_actions3.len(), 6);
//     }
//     #[test]
//     fn test_action_application() {
//         let mut runtime = Runtime::new(100, "../data".into(), None);
//         let behavior = BehaviorDecl {
//             inputs: vec![
//                 ("x".to_string(), Signal::DataFrame(None)),
//                 // ("y".to_string(), Signal::DataFrame(None)),
//             ],
//             output: ("result".to_string(), Signal::DataFrame(None)),
//             operators: Some(vec!["ts_mean".to_string(), "ts_diff".to_string()]),
//             integers: Some(vec![1, 2, 3]),
//             floats: Some(vec![12.3]),
//             strings: Some(vec!["close".to_string()]),
//             weights: None,
//             train: None,
//             supervised_epochs: None,
//         };

//         let _action_space = ActionSpace::from(&behavior);
//         let state = SearchState::from(behavior.clone());
//         let param_values: Vec<Tree> = vec![
//             Tree::Program(test_make_param0()),
//             // Tree::Program(test_make_param1()),
//         ];

//         let (next_state, done) = {
//             let (s0, _d0) = (state, false);
//             let (s1, _d1) = s0.apply_action(Action::ShiftParam(0), &mut runtime, &param_values);
//             let (s2, _d2) = s1.apply_action(Action::ShiftInt(2), &mut runtime, &param_values);
//             let (s3, d3) = s2.apply_action(
//                 Action::Reduce("ts_mean".into()),
//                 &mut runtime,
//                 &param_values,
//             );
//             (s3, d3)
//         };
//         assert!(!done);
//         assert_eq!(next_state.stack.len(), 1);

//         let (_, done) = next_state.apply_action(Action::Done, &mut runtime, &param_values);
//         assert!(done);
//     }
// }
