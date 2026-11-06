use parser::behavior::BehaviorDecl;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use stdlib::OperatorSpec;
use tch::Tensor;

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    ShiftInt(i64),
    ShiftFloat(f64),
    ShiftString(String),
    ShiftParam(usize),
    Reduce(&'static OperatorSpec),
    Done,
}

impl Hash for Action {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Into::<String>::into(self).hash(state);
    }
}

impl Eq for Action {}

impl Into<String> for &Action {
    fn into(self) -> String {
        match self {
            Action::ShiftFloat(x) => format!("{:.8}", x),
            Action::Reduce(x) => format!("!{}", x.name),
            Action::Done => "!done".to_string(),
            Action::ShiftInt(x) => format!("{}", x),
            Action::ShiftString(x) => format!("\"{x}\""),
            Action::ShiftParam(x) => format!("!shift_{}", x),
        }
    }
}

impl From<String> for Action {
    fn from(s: String) -> Self {
        if s == "!done" {
            Action::Done
        } else if let Some(idx) = s.strip_prefix("!shift_") {
            Action::ShiftParam(idx.parse::<usize>().unwrap())
        } else if s.starts_with('"') {
            Action::ShiftString(s.trim_matches('"').to_string())
        } else if let Some(op_name) = s.strip_prefix('!') {
            Action::Reduce(op_name.into())
        } else if let Ok(i) = s.parse::<i64>() {
            Action::ShiftInt(i)
        } else if let Ok(f) = s.parse::<f64>() {
            Action::ShiftFloat(f)
        } else {
            panic!("Unknown action: {}", s);
        }
    }
}

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

    pub fn calculate_mask(&self, valid_actions: &[Action]) -> Tensor {
        let mut mask = vec![false; self.size()];
        for action in valid_actions {
            mask[self.get_idx(action)] = true;
        }
        Tensor::from_slice(&mask)
    }

    pub fn get_class_matrix(&self, device: &tch::Device) -> Tensor {
        let mut matrix = vec![0.0f32; 3 * self.size()];
        for i in 0..self.size() {
            let action = self.get_action(i);
            let class_idx = match action {
                Action::ShiftInt(_) | Action::ShiftFloat(_) | Action::ShiftString(_) => 0,
                Action::ShiftParam(_) => 1,
                Action::Reduce(_) | Action::Done => 2,
            };
            matrix[i * 3 + class_idx] = 1.0;
        }
        Tensor::from_slice(&matrix)
            .view([self.size() as i64, 3])
            .to(*device)
    }
}

impl From<&BehaviorDecl> for ActionSpace {
    fn from(b: &BehaviorDecl) -> ActionSpace {
        let integers = b.integers.clone().unwrap_or_default();
        let floats = b.floats.clone().unwrap_or_default();
        let strings = b.strings.clone().unwrap_or_default();
        let operators = b.operators.clone().unwrap_or_default();

        let integer_offset = 1; // Done
        let float_offset = integer_offset + integers.len();
        let string_offset = float_offset + floats.len();
        let operator_offset = string_offset + strings.len();
        let params_offset = operator_offset + operators.len();
        let num_params = b.inputs.len();

        let mut map = HashMap::new();
        map.insert(0, Action::Done);
        for idx in 1..float_offset {
            map.insert(idx, Action::ShiftInt(integers[idx - 1]));
        }
        for idx in float_offset..string_offset {
            map.insert(idx, Action::ShiftFloat(floats[idx - float_offset]));
        }
        for idx in string_offset..operator_offset {
            map.insert(
                idx,
                Action::ShiftString(strings[idx - string_offset].clone()),
            );
        }
        for idx in operator_offset..params_offset {
            let op: &OperatorSpec = (operators[idx - operator_offset].as_str()).into();
            map.insert(idx, Action::Reduce(op));
        }
        for idx in params_offset..params_offset + num_params {
            map.insert(idx, Action::ShiftParam(idx - params_offset));
        }

        let r_map = HashMap::from_iter(map.iter().map(|(idx, act)| (act.clone(), *idx)));
        ActionSpace { map, r_map }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_consistency() {
        let actions = vec![
            Action::ShiftInt(42),
            Action::ShiftFloat(42.0),
            Action::ShiftString("hello".to_string()),
            Action::ShiftParam(42),
            Action::Reduce("ts_mean".into()),
            Action::Done,
        ];
        for action in actions {
            let action_str: String = (&action).into();
            let action_back: Action = action_str.into();
            assert_eq!(action, action_back);
        }
    }
}
