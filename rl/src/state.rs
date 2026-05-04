use crate::action::{Action, ActionSpace};
use parser::behavior::{BehaviorDecl, FlowDecl, NamedSignal};
use parser::expr::{Expr, Literal};
use runtime::ast::Token;
use runtime::ast::{OperatorSpec, PolishExpr};
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
    pub params: Vec<(String, Signal, bool)>, // true if used. all of them need to be used.
    pub stack: Vec<(Signal, Option<Vec<Vec<f64>>>)>, // (type, expression, data)
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
    pub fn get_valid_actions(&self, action_space: &ActionSpace) -> Vec<Action> {
        todo!()
    }
    pub fn apply_action(self, action: Action) -> (SearchState, bool) {
        //TODO: the elements in stack includes Vec<Vec<f64>> (calculated dataframe), which requires partial calculation of code.
        let done: bool = match action {
            Action::Done => true,
            _ => false,
        };

        let mut next_state = self.clone();
        match action {
            Action::ShiftInt(i) => {
                next_state.expr.push(Token::Literal(Literal::Integer(i)));
                next_state.stack.push((Signal::Int(None), None));
            }
            Action::ShiftFloat(f) => {
                next_state.expr.push(Token::Literal(Literal::Float(f)));
                next_state.stack.push((Signal::Float(None), None));
            }
            Action::ShiftString(s) => {
                next_state.expr.push(Token::Literal(Literal::String(s)));
                next_state.stack.push((Signal::String(None), None));
            }
            Action::Reduce(op) => {
                next_state.stack.push((op.output.clone(), None));
                next_state.expr.push(Token::Operator(op));
            }
            _ => {}
        };

        (next_state, done)
    }
}

#[cfg(test)]
mod tests {
    use parser::behavior::FlowDecl;

    use super::*;

    #[test]
    fn test() {
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

        let action_space = ActionSpace::from(behavior.clone());
        let state = SearchState::from(behavior.clone());

        let (next_state, done) = state.apply_action(Action::ShiftInt(42));
        assert!(!done);
        assert_eq!(next_state.stack.len(), 1);

        let (_, done) = next_state.apply_action(Action::Done);
        assert!(done);
    }
}
