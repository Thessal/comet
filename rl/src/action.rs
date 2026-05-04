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
/// Action
//////////

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    // Increase order (move parameter from unprocessed to stack)
    ShiftInt(i64),
    ShiftFloat(f64),
    ShiftString(String),
    ShiftParam(usize),    // push param into stack
    Reduce(OperatorSpec), // Apply function/behavior and reduce stack
    Done,                 // Successfully matched exit condition
}

impl Hash for Action {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Action::ShiftFloat(x) => (((*x) * 1024.0 * 1024.0).round() as i64).hash(state),
            Action::Reduce(x) => format!("!{}", x.name).hash(state),
            x => x.hash(state),
        }
    }
}

impl Eq for Action {}

impl From<String> for Action {
    fn from(s: String) -> Self {
        if s == "!done" {
            Action::Done
        } else if s.starts_with("!shift_") {
            let idx = s.strip_prefix("!shift_").unwrap();
            Action::ShiftParam(idx.parse::<usize>().unwrap())
        } else if s.starts_with("\"") {
            Action::ShiftString(s.trim_matches('"').to_string())
        } else if let Ok(i) = s.parse::<i64>() {
            Action::ShiftInt(i)
        } else if let Ok(f) = s.parse::<f64>() {
            Action::ShiftFloat(f)
        } else {
            Action::Reduce(OperatorSpec::from(s.as_str()))
        }
    }
}

//////////
/// Action Space
//////////

#[derive(Debug, Clone)]
pub struct ActionSpace {
    map: HashMap<usize, Action>,
    r_map: HashMap<Action, usize>,
}

impl ActionSpace {
    pub fn size(&self) -> usize {
        self.map.len()
    }
    pub fn get_idx(&self, action: &Action) -> usize {
        *self.r_map.get(action).unwrap()
    }
    pub fn get_action(&self, idx: usize) -> Action {
        self.map.get(&idx).unwrap().clone()
    }
}

impl From<BehaviorDecl> for ActionSpace {
    fn from(b: BehaviorDecl) -> ActionSpace {
        let _integers = b.integers.unwrap_or_default();
        let _floats = b.floats.unwrap_or_default();
        let _strings = b.strings.unwrap_or_default();
        let _operators = b.operators.unwrap_or_default();

        let integer_offset = 1; // Done
        let float_offset = integer_offset + _integers.len();
        let string_offset = float_offset + _floats.len();
        let operator_offset = string_offset + _strings.len();
        let params_offset = operator_offset + _operators.len();
        let num_params = b.inputs.len();

        let mut map = HashMap::new();
        map.insert(0, Action::Done);
        for idx in 1..float_offset {
            map.insert(idx, Action::ShiftInt(_integers[idx - 1]));
        }
        for idx in float_offset..string_offset {
            map.insert(idx, Action::ShiftFloat(_floats[idx - float_offset]));
        }
        for idx in string_offset..operator_offset {
            map.insert(
                idx,
                Action::ShiftString(_strings[idx - string_offset].clone()),
            );
        }
        for idx in operator_offset..params_offset {
            let op: OperatorSpec = OperatorSpec::from(_operators[idx - operator_offset].as_str());
            map.insert(idx, Action::Reduce(op));
        }
        for idx in params_offset..params_offset + num_params {
            map.insert(idx, Action::ShiftParam(idx - params_offset));
        }

        let r_map = HashMap::from_iter(map.iter().map(|(idx, act)| (act.clone(), *idx)));
        ActionSpace { map, r_map }
    }
}
