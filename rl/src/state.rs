use crate::action::Action;
use parser::ast::{Network, Node, NodeType};
use parser::expr::Literal;
use stdlib::OperatorSpec;
use stdlib::types::Signal;

//////////
/// Search State
//////////

#[derive(Debug, Clone)]
pub struct AbstractMachine {
    // A signal is represented as a root node in callgraph, and an item in stack.
    // 1. When you introduce a function parameter, you add a new item in stack.
    //    Add address of the node in addr. (add a node in callgraph if it is not already in the graph.)
    // 2. When you reduce an operator, merge root nodes in callgraph into a single node.
    //    Pop the arguments from stack and addr. Push the output signal to stack.
    params: Vec<(Signal, usize)>, // parameter indices of the function. maps parameter order into address.
    // params need to be initialized when AbstractMachine is created from SearchState::from(Network)
    stack: Vec<(Signal, usize)>,   // stack (types, address)
    pub callgraph: Network,        // memory
    callgraph_behavior_idx: usize, // location of the behavior node in callgraph
}

impl AbstractMachine {
    pub fn get_stack(&self) -> (&Vec<(Signal, usize)>, &Network) {
        (&self.stack, &self.callgraph)
    }

    pub fn check_reduce(&self, operator_spec: &OperatorSpec) -> bool {
        // type checking
        assert!(operator_spec.inputs.len() <= self.stack.len());
        for (op_input, stack_item) in operator_spec.inputs.iter().zip(self.stack.iter().rev()) {
            if std::mem::discriminant(op_input) != std::mem::discriminant(&stack_item.0) {
                return false;
            }
        }
        true
    }
    fn get_param_addr(&mut self, action: &Action) -> Option<usize> {
        // Modifies only callgraph
        // write data to callgraph(memory) if not exist
        match &action {
            Action::ShiftInt(i) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::Integer(*i))),
            ),
            Action::ShiftFloat(f) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::Float(*f))),
            ),
            Action::ShiftString(s) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::String(s.clone()))),
            ),
            Action::ShiftParam(p) => {
                let (_signal_type, addr) = &self.params[*p];
                Some(*addr)
            }
            Action::Reduce(op) => {
                let idx = self.callgraph.add_node(NodeType::Operator(op.clone()));
                let arity = op.inputs.len();
                for (_, addr) in self.stack.iter().rev().take(arity) {
                    // or is it self.stack.iter().skip(stack_len - arity) ?
                    self.callgraph.add_child(idx, *addr);
                }
                Some(idx)
            }
            Action::Done => None,
        }
    }

    // addr and callgraph update
    fn push(&mut self, action: &Action) {
        let addr_incoming = self.get_param_addr(action);

        // Consume stack
        if let Action::Reduce(op) = action {
            assert!(self.check_reduce(op));
            self.stack.truncate(self.stack.len() - op.inputs.len());
        }

        // Push addr_incoming to stack
        let signal_incoming = self.action_to_signal(action);
        if let Some(addr) = addr_incoming {
            self.stack.push((signal_incoming, addr));
        };
    }

    fn action_to_signal(&self, action: &Action) -> Signal {
        match action {
            Action::ShiftInt(i) => Signal::Int(Some(*i)),
            Action::ShiftFloat(f) => Signal::Float(Some(*f)),
            Action::ShiftString(s) => Signal::String(Some(s.clone())),
            Action::ShiftParam(i) => self.params[*i].0.clone(),
            Action::Reduce(op_spec) => op_spec.output_shape.clone(),
            Action::Done => unreachable!(),
        }
    }
}

impl From<Network> for AbstractMachine {
    fn from(callgraph: Network) -> Self {
        let (behavior_idx, behavior_decl) = callgraph.get_behavior();
        Self {
            params: behavior_decl
                .inputs
                .iter()
                .zip(callgraph.nodes[behavior_idx].children.iter())
                .map(|(sig, &addr)| (sig.clone(), addr))
                .collect(),
            stack: vec![],
            callgraph: callgraph.clone(),
            callgraph_behavior_idx: behavior_idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchState {
    pub machine: AbstractMachine, // used to track arguments and actions of episode
    orig_callgraph_size: usize,
    orig_behavior_node: Node,
}

impl SearchState {
    pub fn new(callgraph: &Network) -> Self {
        let machine: AbstractMachine = callgraph.clone().into();
        let behavior_idx = machine.callgraph_behavior_idx;
        let orig_behavior_node = callgraph.nodes[behavior_idx].clone();
        Self {
            machine,
            orig_callgraph_size: callgraph.nodes.len(),
            orig_behavior_node,
        }
    }

    pub fn reset(&mut self) {
        // clear the stack to avoid leftover nodes from previous trajectories
        self.machine.stack.clear();
        // drop appended nodes
        self.machine
            .callgraph
            .nodes
            .truncate(self.orig_callgraph_size);
        // restore behavior node
        self.machine.callgraph.nodes[self.machine.callgraph_behavior_idx] =
            self.orig_behavior_node.clone();
    }

    pub fn apply_action(&mut self, action: &Action) {
        match action {
            Action::Done => {
                // Check stack size
                assert!(self.machine.stack.len() == 1);
                // do the rewiring to replace behavior node with the subgraph
                // 1. get subgraph address
                let (subgraph_output_type, subgraph_root_idx) = self.machine.stack.last().unwrap();
                assert!(matches!(subgraph_output_type, Signal::DataFrame(_)));
                // 2. replace behavior node with subgraph root
                let root_node = self.machine.callgraph.nodes[*subgraph_root_idx].clone();
                self.machine.callgraph.nodes[self.machine.callgraph_behavior_idx] = root_node;
            }
            a => {
                self.machine.push(a);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use parser::behavior::BehaviorDecl;
    use runtime::runtime::Runtime;
    use stdlib::types::Signal;
    #[test]
    fn test_action_apply_reduce() {
        todo!()
    }
    #[test]
    fn test_get_valid_actions() {}
}
