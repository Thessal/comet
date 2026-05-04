use crate::action::{Action, ActionSpace};
use burn::prelude::Backend;
use burn::tensor::TensorData;
use parser::behavior::{BehaviorDecl, FlowDecl, NamedSignal};
use parser::expr::{Expr, Literal};
use runtime::ast::{OperatorSpec, PolishExpr, Program};
use runtime::ast::{Token, Tree};
use runtime::runtime::Runtime;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use stdlib::types::Signal;

//////////
/// Search State
//////////

#[derive(Debug, Clone)]
pub struct SearchState {
    pub params: Vec<(String, Signal, bool)>, // (param name, param data, is_used). all params need to be used in the end.
    pub stack: Vec<(PolishExpr, Tree, Signal)>, // polish expr, tree, data
    pub expr: PolishExpr,                    // Polish expression (added for convenience)
}

impl From<BehaviorDecl> for SearchState {
    fn from(x: BehaviorDecl) -> Self {
        SearchState {
            params: x
                .inputs
                .iter()
                .map(|(name, signal)| (name.clone(), signal.clone(), false))
                .collect(),
            stack: vec![],
            expr: PolishExpr::new(),
        }
    }
}

impl SearchState {
    pub fn get_valid_actions(&self, action: &ActionSpace) -> Vec<Action> {
        todo!()
    }
    fn make_tree(
        &self,
        action: Action,
        runtime: &mut Runtime,
    ) -> (SearchState, PolishExpr, Tree, Signal) {
        let mut next_state = self.clone();
        let (expr, tree, data): (PolishExpr, Tree, Signal) = match action {
            Action::ShiftInt(i) => (
                vec![Token::Literal(Literal::Integer(i))],
                Tree::Literal(Literal::Integer(i)),
                Signal::Int(Some(i)),
            ),
            Action::ShiftFloat(f) => (
                vec![Token::Literal(Literal::Float(f))],
                Tree::Literal(Literal::Float(f)),
                Signal::Float(Some(f)),
            ),
            Action::ShiftString(s) => (
                vec![Token::Literal(Literal::String(s.clone()))],
                Tree::Literal(Literal::String(s.clone())),
                Signal::String(Some(s.clone())),
            ),
            Action::ShiftParam(i) => {
                let (param_name, signal, _) = next_state.params[i].clone();
                next_state.params[i].2 = true;
                (
                    vec![Token::Parameter(i)],
                    Tree::Program(Program {
                        spec: OperatorSpec {
                            name: format!("!shift_{}", param_name),
                            inputs_type: vec![],
                            output_type: signal.clone(),
                        },
                        polish_expression: Some(PolishExpr::from(vec![Token::Parameter(i)])),
                        parameters: None,
                    }),
                    signal,
                )
            }
            Action::Reduce(op) => {
                let arity = op.inputs_type.len();
                let inputs = next_state.stack.split_off(next_state.stack.len() - arity);
                let exprs: Vec<PolishExpr> = inputs.iter().map(|x| x.0.clone()).collect();
                let trees: Vec<Tree> = inputs.iter().map(|x| x.1.clone()).collect();
                let datas: Vec<Signal> = inputs.iter().map(|x| x.2.clone()).collect();

                assert!(datas.len() == arity);
                assert!(
                    datas
                        .iter()
                        .zip(op.inputs_type.iter())
                        .all(|(d, i)| { std::mem::discriminant(d) == std::mem::discriminant(i) })
                );

                let expr: PolishExpr = exprs
                    .into_iter()
                    .chain(vec![vec![Token::Operator(op.clone())]].into_iter())
                    .flatten()
                    .collect();

                let tree: Tree = Tree::Program(Program {
                    spec: op,
                    polish_expression: Some(expr.clone()),
                    parameters: Some(trees),
                });
                let data = runtime.run(&tree);
                (expr, tree, data)
            }
            Action::Done => panic!("Done action should be handled in prior step."),
        };
        (next_state, expr, tree, data)
    }

    pub fn apply_action(&mut self, action: Action, runtime: &mut Runtime) -> (SearchState, bool) {
        match action {
            Action::Done => {
                let next_state = self.clone();
                let params_consumed: bool = next_state.params.iter().all(|(_, _, used)| *used);
                let stack_size: bool = next_state.stack.len() == 1;
                assert!(params_consumed && stack_size);
                (next_state, true)
            }
            x => {
                let (mut next_state, expr, tree, data) = self.make_tree(x, runtime);
                next_state.stack.push((expr, tree, data));
                (next_state, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use parser::behavior::FlowDecl;

    use super::*;

    #[test]
    fn test() {
        let mut runtime = Runtime::new(100, "../data".into());
        let behavior = BehaviorDecl {
            inputs: vec![
                ("x".to_string(), Signal::DataFrame(None)),
                ("y".to_string(), Signal::DataFrame(None)),
            ],
            output: ("result".to_string(), Signal::DataFrame(None)),
            operators: Some(vec!["ts_mean".to_string(), "ts_diff".to_string()]),
            integers: Some(vec![1, 2, 3]),
            floats: Some(vec![12.3]),
            strings: Some(vec!["close".to_string()]),
            weights: None,
            train: None,
            supervised_epochs: None,
        };

        let action_space = ActionSpace::from(&behavior);
        let mut state = SearchState::from(behavior.clone());

        let (mut next_state, done) = state.apply_action(Action::ShiftInt(42), &mut runtime);
        assert!(!done);
        assert_eq!(next_state.stack.len(), 1);

        let (_, done) = next_state.apply_action(Action::Done, &mut runtime);
        assert!(done);
    }
}
