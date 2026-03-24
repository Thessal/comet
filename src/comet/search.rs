use crate::comet::ast::{Ident, TypeDecl};

#[derive(Debug, Clone, PartialEq)]
pub struct SearchState {
    pub unprocessed_params: Vec<TypeDecl>,
    pub stack: Vec<TypeDecl>,
    pub sequence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    Shift,         // Increase order (move parameter from unprocessed to stack)
    Reduce(Ident), // Apply function/behavior and reduce stack
    Done,          // Successfully matched exit condition
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchEnv {
    pub target_return: TypeDecl,
}

impl SearchEnv {
    pub fn new(target_return: TypeDecl) -> Self {
        SearchEnv { target_return }
    }

    pub fn get_valid_actions(
        &self,
        state: &SearchState,
        available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    ) -> Vec<Action> {
        let mut actions = Vec::new();

        if state.unprocessed_params.is_empty()
            && state.stack.len() == 1
            && state.stack[0] == self.target_return
        {
            actions.push(Action::Done);
        }

        if !state.unprocessed_params.is_empty() {
            actions.push(Action::Shift);
        }

        for (name, param_types, _ret_type) in available_funcs {
            // Check if stack suffix matches param_types
            if param_types.is_empty() {
                // Zero-ary operator - can always apply
                actions.push(Action::Reduce(name.clone()));
            } else if state.stack.len() >= param_types.len() {
                let stack_suffix = &state.stack[state.stack.len() - param_types.len()..];
                if stack_suffix == param_types.as_slice() {
                    actions.push(Action::Reduce(name.clone()));
                }
            }
        }

        actions
    }

    pub fn step(
        &self,
        state: &SearchState,
        action: Action,
        available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    ) -> Result<SearchState, String> {
        let mut new_state = state.clone();

        match action {
            Action::Shift => {
                if let Some(param) = new_state.unprocessed_params.pop() {
                    new_state.stack.push(param);
                    Ok(new_state)
                } else {
                    Err("Cannot shift: no unprocessed parameters remaining.".to_string())
                }
            }
            Action::Reduce(func_name) => {
                let func_def = available_funcs
                    .iter()
                    .find(|(n, _, _)| *n == func_name)
                    .ok_or_else(|| format!("Unknown function: {}", func_name))?;

                let param_types = &func_def.1;
                let ret_type = &func_def.2;

                if new_state.stack.len() < param_types.len() {
                    return Err(format!("Cannot reduce {}: stack too small.", func_name));
                }

                // Check types and pop
                for expected_type in param_types.iter().rev() {
                    let actual = new_state.stack.pop().unwrap();
                    if &actual != expected_type {
                        return Err(format!(
                            "Type mismatch reducing {}: expected {:?}, got {:?}",
                            func_name, expected_type, actual
                        ));
                    }
                }

                new_state.stack.push(ret_type.clone());

                // Add to sequence prefix explicitly for tracking (simulated postfix order for now,
                // but real serialization will build AST and convert using `dag_to_sequence`).
                new_state.sequence.push(func_name.clone());

                Ok(new_state)
            }
            Action::Done => {
                if new_state.unprocessed_params.is_empty()
                    && new_state.stack.len() == 1
                    && new_state.stack[0] == self.target_return
                {
                    Ok(new_state)
                } else {
                    Err("Exit condition not met.".to_string())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_behavior_search_logic() {
        // From docs/behavior_search.md example
        // Input params: [DataFrame, DataFrame, Float, DataFrame] (reversed since we pop from end: pop D first, then F, etc. Wait!
        // Actually, the example in markdown says:
        // Input: [D, D, F, D]. Increase order -> [D, D, F], [D]
        // This means it pops the LAST element.
        let mut state = SearchState {
            unprocessed_params: vec![
                TypeDecl::DataFrame,
                TypeDecl::DataFrame,
                TypeDecl::Float,
                TypeDecl::DataFrame,
            ],
            stack: vec![],
            sequence: vec![],
        };

        let env = SearchEnv::new(TypeDecl::DataFrame); // Target return: DataFrame

        let available_funcs = vec![
            (
                "consume_float".to_string(),
                vec![TypeDecl::Float],
                TypeDecl::Void,
            ),
            (
                "cs_rank".to_string(),
                vec![TypeDecl::DataFrame],
                TypeDecl::DataFrame,
            ),
            (
                "ts_diff".to_string(),
                vec![TypeDecl::DataFrame, TypeDecl::Float],
                TypeDecl::DataFrame,
            ),
            (
                "divide".to_string(),
                vec![TypeDecl::DataFrame, TypeDecl::DataFrame],
                TypeDecl::DataFrame,
            ),
        ];

        // 1st step: Shift
        state = env.step(&state, Action::Shift, &available_funcs).unwrap();
        assert_eq!(
            state.unprocessed_params,
            vec![TypeDecl::DataFrame, TypeDecl::DataFrame, TypeDecl::Float]
        );
        assert_eq!(state.stack, vec![TypeDecl::DataFrame]);

        // 2nd step: cs_rank
        state = env
            .step(
                &state,
                Action::Reduce("cs_rank".to_string()),
                &available_funcs,
            )
            .unwrap();
        assert_eq!(
            state.unprocessed_params,
            vec![TypeDecl::DataFrame, TypeDecl::DataFrame, TypeDecl::Float]
        );
        assert_eq!(state.stack, vec![TypeDecl::DataFrame]);

        // 3rd step: Shift
        state = env.step(&state, Action::Shift, &available_funcs).unwrap();
        assert_eq!(
            state.unprocessed_params,
            vec![TypeDecl::DataFrame, TypeDecl::DataFrame]
        );
        assert_eq!(state.stack, vec![TypeDecl::DataFrame, TypeDecl::Float]);

        // 4th step: Shift
        state = env.step(&state, Action::Shift, &available_funcs).unwrap();
        assert_eq!(state.unprocessed_params, vec![TypeDecl::DataFrame]);
        assert_eq!(
            state.stack,
            vec![TypeDecl::DataFrame, TypeDecl::Float, TypeDecl::DataFrame]
        );

        // Validate possible actions
        let actions = env.get_valid_actions(&state, &available_funcs);
        assert!(actions.contains(&Action::Shift));
        assert!(actions.contains(&Action::Reduce("cs_rank".to_string())));
        assert!(actions.contains(&Action::Reduce("ts_diff".to_string())));
    }
}
