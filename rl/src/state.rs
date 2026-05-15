use crate::action::{Action, ActionSpace};

use parser::behavior::BehaviorDecl;
use parser::expr::Literal;
use runtime::ast::{OperatorSpec, PolishExpr, Program};
use runtime::ast::{Token, Tree};
use runtime::runtime::Runtime;
use stdlib::types::Signal;

//////////
/// Search State
//////////

#[derive(Debug, Clone)]
pub struct SearchState {
    pub params: Vec<(String, Signal, bool)>, // (param name, param type, is_used). all params need to be used in the end.
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
    pub fn is_done_valid(&self) -> (bool, bool, bool) {
        let params_consumed: bool = self.params.iter().all(|(_, _, used)| *used);
        let stack_size: bool = self.stack.len() == 1;
        let is_done_valid: bool = params_consumed && stack_size;
        (is_done_valid, params_consumed, stack_size)
    }
    pub fn get_valid_actions(&self, action: &ActionSpace) -> Vec<Action> {
        let stack_type_and_data: Vec<Signal> =
            self.stack.iter().map(|(_, _, s)| s.clone()).collect();
        let mut valid_actions: Vec<Action> = vec![];
        for i in 0..action.size() {
            let a = action.get_action(i);
            let is_valid: bool = match a {
                Action::Reduce(OperatorSpec {
                    name: _name,
                    inputs_type,
                    output_type: _output_type,
                }) => {
                    if inputs_type.len() > self.stack.len() {
                        false
                    } else {
                        inputs_type
                            .iter()
                            .rev()
                            .zip(stack_type_and_data.iter().rev())
                            // compare variant only. (not data)
                            .all(|(i, s)| std::mem::discriminant(i) == std::mem::discriminant(s))
                    }
                }
                Action::Done => self.is_done_valid().0,
                _ => self.stack.len() < 5, // allow introducing variables until stack depth 5
                                           // TODO: adjust
            };
            if is_valid {
                valid_actions.push(action.get_action(i));
            }
        }
        valid_actions
    }
    fn make_tree(
        &self,
        action: Action,
        param_values: &Vec<Tree>,
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
                let (_param_name, _output_type, _used) = next_state.params[i].clone();
                next_state.params[i].2 = true;
                let tree = param_values[i].clone();
                (vec![Token::Parameter(i)], tree.clone(), runtime.run(&tree))
            }
            Action::Reduce(op) => {
                let arity = op.inputs_type.len();
                let inputs = next_state.stack.split_off(next_state.stack.len() - arity);
                let exprs: Vec<PolishExpr> = inputs.iter().map(|x| x.0.clone()).collect();
                let trees: Vec<Tree> = inputs.iter().map(|x| x.1.clone()).collect();
                let datas: Vec<Signal> = inputs.iter().map(|x| x.2.clone()).collect();

                assert!(datas.len() == arity);
                // operator arguments type check - RPN
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

    pub fn apply_action(
        &self,
        action: Action,
        runtime: &mut Runtime,
        param_values: &Vec<Tree>,
    ) -> (SearchState, bool) {
        match action {
            Action::Done => {
                let next_state = self.clone();
                let (is_done_valid, _params_consumed, _stack_size) = next_state.is_done_valid();
                assert!(is_done_valid);
                (next_state, true)
            }
            other_action => {
                let token = match &other_action {
                    Action::ShiftInt(i) => Token::Literal(Literal::Integer(*i)),
                    Action::ShiftFloat(f) => Token::Literal(Literal::Float(*f)),
                    Action::ShiftString(s) => Token::Literal(Literal::String(s.clone())),
                    Action::ShiftParam(i) => Token::Parameter(*i),
                    Action::Reduce(op) => Token::Operator(op.clone()),
                    Action::Done => unreachable!(),
                };
                let (mut next_state, expr, tree, data) =
                    self.make_tree(other_action, param_values, runtime);

                next_state.expr.push(token);
                next_state.stack.push((expr, tree, data));
                (next_state, false)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use runtime::runtime::test_make_param0;

    use super::*;
    #[test]
    fn test_get_valid_actions() {
        let mut runtime = Runtime::new(100, "../data".into());
        let behavior = BehaviorDecl {
            inputs: vec![("x".to_string(), Signal::DataFrame(None))],
            output: ("result".to_string(), Signal::DataFrame(None)),
            operators: Some(vec!["ts_mean".to_string()]),
            integers: Some(vec![0, 1]),
            floats: Some(vec![0.0, 3.0]),
            strings: None,
            weights: None,
            train: None,
            supervised_epochs: None,
        };
        let action_space: ActionSpace = (&behavior).into();
        let state = SearchState::from(behavior);
        let valid_actions = state.get_valid_actions(&action_space);
        assert_eq!(valid_actions.len(), 5); //[ShiftInt(0), ShiftInt(1), ShiftFloat(0.0), ShiftFloat(3.0), ShiftParam(0)]
        assert!(valid_actions.contains(&Action::ShiftInt(0)));
        assert!(!valid_actions.contains(&Action::ShiftInt(-1)));
        assert!(valid_actions.contains(&Action::ShiftFloat(0.0)));
        assert!(!valid_actions.contains(&Action::ShiftFloat(-1.0)));

        let param_values: Vec<Tree> = vec![Tree::Program(test_make_param0())];
        let (state, _1) = state.apply_action(Action::ShiftParam(0), &mut runtime, &param_values);
        let (state, _1) = state.apply_action(Action::ShiftInt(0), &mut runtime, &param_values);
        // println!("{:?}", state.stack);
        // println!("action_space {:?}", action_space);
        let valid_actions2 = state.get_valid_actions(&action_space);
        // println!("valid_actions2 {:?}", valid_actions2);
        assert!(valid_actions2.contains(&Action::Reduce(OperatorSpec::from("ts_mean"))));
        assert_eq!(valid_actions2.len(), 6); // [ShiftInt(0), ShiftInt(1), ShiftFloat(0.0), ShiftFloat(3.0), Reduce("ts_mean"), ShiftParam(0)]

        let operator_offset = 1 + 2 + 2 + 0; // Action space layout [done, integers, floats, strings, operators, params]
        let action = action_space.get_action(operator_offset);
        assert!(matches!(action, Action::Reduce(_)));
        let (state, _1) = state.apply_action(action, &mut runtime, &param_values);
        let valid_actions3 = state.get_valid_actions(&action_space);
        // println!(
        //     "is_done_valid: is_done_valid, params_consumed, stack_size = {:?}",
        //     state.is_done_valid()
        // );
        // println!("stack_size: {:?}", state.stack.len());
        // println!("valid_actions3 {:?}", valid_actions3);
        assert!(valid_actions3.contains(&Action::Done));
        assert_eq!(valid_actions3.len(), 6);
    }
    #[test]
    fn test_action_application() {
        let mut runtime = Runtime::new(100, "../data".into());
        let behavior = BehaviorDecl {
            inputs: vec![
                ("x".to_string(), Signal::DataFrame(None)),
                // ("y".to_string(), Signal::DataFrame(None)),
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

        let _action_space = ActionSpace::from(&behavior);
        let state = SearchState::from(behavior.clone());
        let param_values: Vec<Tree> = vec![
            Tree::Program(test_make_param0()),
            // Tree::Program(test_make_param1()),
        ];

        let (next_state, done) = {
            let (s0, _d0) = (state, false);
            let (s1, _d1) = s0.apply_action(Action::ShiftParam(0), &mut runtime, &param_values);
            let (s2, _d2) = s1.apply_action(Action::ShiftInt(2), &mut runtime, &param_values);
            let (s3, d3) = s2.apply_action(
                Action::Reduce("ts_mean".into()),
                &mut runtime,
                &param_values,
            );
            (s3, d3)
        };
        assert!(!done);
        assert_eq!(next_state.stack.len(), 1);

        let (_, done) = next_state.apply_action(Action::Done, &mut runtime, &param_values);
        assert!(done);
    }
}
