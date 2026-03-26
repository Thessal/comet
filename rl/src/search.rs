use parser::program::{Ident, TypeDecl};

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
                    new_state.sequence.push("!shift".to_string());
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

                // Only push if not Void
                if *ret_type != TypeDecl::Void {
                    new_state.stack.push(ret_type.clone());
                }

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

use runtime::runtime::Runtime;
use stdlib::ParamType;

#[derive(Clone, Debug)]
pub struct EvaluatedSample {
    pub actions: Vec<String>,
    pub fitness: Vec<f64>,
}

pub fn generate_top_k_samples(
    behavior: &parser::program::BehaviorDecl,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    top_k: usize,
) -> Vec<EvaluatedSample> {
    let num_samples = behavior.supervised_samples.unwrap_or(100);
    let target_type = behavior.return_type.clone();

    let unprocessed_params: Vec<TypeDecl> = behavior
        .args
        .iter()
        .rev()
        .map(|arg| arg.type_decl.clone())
        .collect();

    let param_names: Vec<String> = behavior
        .args
        .iter()
        .rev()
        .map(|arg| {
            if arg.type_decl == TypeDecl::Float {
                "1.0".to_string()
            } else {
                "volume".to_string() // guaranteed to exist from DataManager mocks internally natively mapped to Variable("volume")
            }
        })
        .collect();

    let initial_state = SearchState {
        unprocessed_params,
        stack: vec![],
        sequence: vec![],
    };

    let mut samples = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    let env = SearchEnv::new(target_type);
    let mut runtime = Runtime::new(0, "data");

    let mut sample_count = 0;
    let mut attempts = 0;
    while sample_count < num_samples && attempts < 1000 {
        print!("Attempt: {}\r", attempts);
        attempts += 1;
        let mut current_state = initial_state.clone();
        let mut hit_done = false;

        for step in 0..64 {
            let valid_actions = env.get_valid_actions(&current_state, available_funcs);

            if valid_actions.contains(&Action::Done) && step >= 5 {
                current_state = env
                    .step(&current_state, Action::Done, available_funcs)
                    .unwrap();
                hit_done = true;
                break;
            } else if valid_actions.contains(&Action::Done) && valid_actions.len() == 1 {
                current_state = env
                    .step(&current_state, Action::Done, available_funcs)
                    .unwrap();
                hit_done = true;
                break;
            }

            let filtered_actions: Vec<&Action> = valid_actions
                .iter()
                .filter(|a| **a != Action::Done)
                .collect();

            if filtered_actions.is_empty() {
                break;
            }

            use rand::seq::SliceRandom;
            let action = filtered_actions.choose(&mut rng).unwrap();
            current_state = env
                .step(&current_state, (*action).clone(), available_funcs)
                .unwrap();
        }

        if hit_done {
            let fitness =
                match runtime.evaluate_sequence(&current_state.sequence, param_names.clone()) {
                    Ok(ParamType::DataFrame(output)) => {
                        runtime::fitness::evaluate_fitness(&mut runtime.dmgr, &output)
                    }
                    _ => vec![-1.0], // Penalize runtime failure
                };

            samples.push(EvaluatedSample {
                actions: current_state.sequence.clone(),
                fitness,
            });
            sample_count += 1;
        }
    }
    println!();

    // Sort descending by fitness
    samples.sort_by(|a, b| {
        let f_a = a.fitness.first().copied().unwrap_or(-1.0);
        let f_b = b.fitness.first().copied().unwrap_or(-1.0);
        f_b.partial_cmp(&f_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    if samples.len() > top_k {
        samples.truncate(top_k);
    }

    samples
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

        // Validate possible actions BEFORE 4th shift
        let actions_before_4th = env.get_valid_actions(&state, &available_funcs);
        assert!(actions_before_4th.contains(&Action::Shift));
        assert!(actions_before_4th.contains(&Action::Reduce("ts_diff".to_string())));

        // 4th step: Shift
        state = env.step(&state, Action::Shift, &available_funcs).unwrap();
        assert_eq!(state.unprocessed_params, vec![TypeDecl::DataFrame]);
        assert_eq!(
            state.stack,
            vec![TypeDecl::DataFrame, TypeDecl::Float, TypeDecl::DataFrame]
        );

        // Validate possible actions AFTER 4th shift
        let actions = env.get_valid_actions(&state, &available_funcs);
        assert!(actions.contains(&Action::Shift));
        assert!(actions.contains(&Action::Reduce("cs_rank".to_string())));
    }

    #[test]
    fn test_consume_minimal_integration() {
        use parser::parser::parse;
        use parser::program::{BehaviorDecl, Declaration};

        let src = std::fs::read_to_string("../examples/behavior_1.cm").expect("Read failed");
        let program = parse(&src).expect("Failed to parse behavior_1.cm");

        let available_funcs = crate::env::get_available_funcs();

        let mut target_behavior = None;
        for decl in &program.declarations {
            if let Declaration::Behavior(b) = decl {
                if b.name == "Comparator" {
                    target_behavior = Some(b.clone());
                }
            }
        }

        let behavior = target_behavior.expect("Failed to find Comparator behavior");

        println!("Inferring action candidates from the code:");
        for f in &available_funcs {
            println!("- {}({:?}) -> {:?}", f.0, f.1, f.2);
        }

        println!("\nGenerating sample expression trees and evaluating using runtime...");
        let samples = generate_top_k_samples(&behavior, &available_funcs, 3);

        for (i, sample) in samples.iter().enumerate() {
            println!("Sample {}: Fitness = {:?}", i + 1, sample.fitness);
            println!("  Actions: {:?}", sample.actions);
        }

        assert!(
            !samples.is_empty(),
            "Should generate at least one valid expression tree"
        );
    }
}
