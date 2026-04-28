use crate::action::{Action, ActionSpace};
use parser::expr::{Expr, Literal};
use parser::program::{BehaviorDecl, FlowDecl, NamedSignal};
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
    pub fn apply_action(self, action: Action) -> (SearchState, bool) {
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
    use parser::program::FlowDecl;

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

        let action_space = ActionSpace::from(behavior);
        let state = SearchState::from(behavior);

        let (next_state, done) = state.apply_action(Action::ShiftInt(42));
        assert!(!done);
        assert_eq!(next_state.stack.len(), 1);

        let (_, done) = next_state.apply_action(Action::Done);
        assert!(done);
    }
}

// #[derive(Clone, Debug)]
// pub struct EvaluatedSample {
//     pub actions: Vec<String>,
//     pub fitness: Vec<f64>,
// }

// pub fn generate_top_k_samples(
//     behavior: &parser::program::BehaviorDecl,
//     available_funcs: &[runtime::ast::OperatorSpec],
//     top_k: usize,
//     selection_rule: fn(&Vec<f64>) -> bool,
//     runtime: &mut Runtime,
// ) -> Vec<EvaluatedSample> {
//     let num_samples = top_k;
//     let target_type = behavior.return_type.clone();

//     let unprocessed_params: Vec<TypeDecl> = behavior
//         .args
//         .iter()
//         .rev()
//         .map(|arg| arg.type_decl.clone())
//         .collect();

//     let param_names: Vec<String> = behavior
//         .args
//         .iter()
//         .rev()
//         .map(|arg| {
//             if arg.type_decl == TypeDecl::Float {
//                 "1.0".to_string()
//             } else {
//                 "volume".to_string() // guaranteed to exist from DataManager mocks internally natively mapped to Variable("volume")
//             }
//         })
//         .collect();

//     let initial_state = SearchState {
//         unprocessed_params,
//         stack: vec![],
//         sequence: vec![],
//     };

//     let mut samples = Vec::with_capacity(num_samples);
//     let mut rng = rand::thread_rng();

//     let env = SearchEnv::new(
//         target_type,
//         behavior.integers.clone().unwrap_or_default(),
//         behavior.floats.clone().unwrap_or_default(),
//         behavior.strings.clone().unwrap_or_default(),
//         true,
//     );

//     let mut attempts = 0;

//     // 1. Generate a large pool of structurally valid sequences first
//     let target_pool = num_samples * 20; // Ensure a sizable target objective portfolio context
//     let mut structurally_valid_sequences = std::collections::HashSet::new();

//     while structurally_valid_sequences.len() < target_pool && attempts < 50000 * top_k {
//         print!(
//             "Attempt: {} | Pool: {}/{}\r",
//             attempts,
//             structurally_valid_sequences.len(),
//             target_pool
//         );
//         let _ = std::io::stdout().flush();
//         attempts += 1;

//         let mut current_state = initial_state.clone();
//         let minimal_len = initial_state.unprocessed_params.len() + 1;
//         let mut hit_done = false;

//         for step in 0..64 {
//             let valid_actions = env.get_valid_actions(&current_state, available_funcs);

//             if valid_actions.contains(&Action::Done) && step >= minimal_len {
//                 current_state = env
//                     .step(&current_state, Action::Done, available_funcs)
//                     .unwrap();
//                 hit_done = true;
//                 break;
//             } else if valid_actions.contains(&Action::Done) && valid_actions.len() == 1 {
//                 current_state = env
//                     .step(&current_state, Action::Done, available_funcs)
//                     .unwrap();
//                 hit_done = true;
//                 break;
//             }

//             let filtered_actions: Vec<&Action> = valid_actions
//                 .iter()
//                 .filter(|a| **a != Action::Done)
//                 .collect();

//             if filtered_actions.is_empty() {
//                 break;
//             }

//             use rand::seq::SliceRandom;
//             let action = filtered_actions.choose(&mut rng).unwrap();
//             current_state = env
//                 .step(&current_state, (*action).clone(), available_funcs)
//                 .unwrap();
//         }

//         if hit_done {
//             structurally_valid_sequences.insert(current_state.sequence);
//         }
//     }
//     println!(
//         "\nGenerated {} structurally valid sequences",
//         structurally_valid_sequences.len()
//     );

//     let structurally_valid_sequences: Vec<_> = structurally_valid_sequences.into_iter().collect();

//     // 2. Evaluate all valid structure sequences sequentially
//     let mut parsed_outputs = Vec::new();
//     let mut valid_indices = Vec::new();
//     for (i, seq) in structurally_valid_sequences.iter().enumerate() {
//         match runtime.evaluate_sequence(seq, param_names.clone()) {
//             Ok(stdlib::Signal::DataFrame(output)) => {
//                 parsed_outputs.push(output);
//                 valid_indices.push(i);
//             }
//             Ok(x) => {
//                 println!("Wrong data type was returned: {:?}", x);
//             }
//             Err(e) => {
//                 println!("Failed to evaluate sequence: {:?} ({})", seq, e);
//             }
//         }
//     }

//     // 3. Batch evaluating native portfolio math (marginal Value-Added Sharpe combinations)
//     let valid_refs: Vec<&[Vec<f64>]> = parsed_outputs.iter().map(|o| o.as_slice()).collect();
//     let batch_fitness =
//         runtime::fitness::evaluate_fitness_batch_add_value(&mut runtime.dmgr, &valid_refs);

//     let mut fitness_scores = vec![vec![-1.0]; structurally_valid_sequences.len()];
//     for (idx, metrics) in valid_indices.into_iter().zip(batch_fitness.into_iter()) {
//         fitness_scores[idx] = vec![runtime::fitness::fitness_summary(&metrics)];
//     }

//     // 4. Construct samples structurally and securely filter mapping mathematically
//     for (i, seq) in structurally_valid_sequences.into_iter().enumerate() {
//         let fitness = fitness_scores[i].clone();
//         if selection_rule(&fitness) {
//             samples.push(EvaluatedSample {
//                 actions: seq,
//                 fitness,
//             });
//         }
//     }

//     // 5. Sort descending by fitness
//     samples.sort_by(|a, b| {
//         let f_a = a.fitness.first().copied().unwrap_or(-1.0);
//         let f_b = b.fitness.first().copied().unwrap_or(-1.0);
//         f_b.partial_cmp(&f_a).unwrap_or(std::cmp::Ordering::Equal)
//     });

//     if samples.len() > top_k {
//         samples.truncate(top_k);
//     }
//     //TODO: drop duplicates

//     samples
// }

// impl SearchEnv {
//     pub fn new(
//         target_return: TypeDecl,
//         available_integers: Vec<i64>,
//         available_floats: Vec<f64>,
//         available_strings: Vec<String>,
//         limit_introducing_constants: bool,
//     ) -> Self {
//         SearchEnv {
//             target_return,
//             available_integers,
//             available_floats,
//             available_strings,
//             limit_introducing_constants_too_much: limit_introducing_constants,
//         }
//     }

//     pub fn get_valid_actions(
//         &self,
//         state: &SearchState,
//         available_funcs: &[runtime::ast::OperatorSignature],
//     ) -> Vec<Action> {
//         let mut actions = Vec::new();

//         if state.unprocessed_params.is_empty()
//             && state.stack.len() == 1
//             && state.stack[0] == self.target_return
//         {
//             actions.push(Action::Done);
//         }

//         if !state.unprocessed_params.is_empty() {
//             actions.push(Action::Shift);
//         }

//         // If limit_introducing_constants_too_much is true, prevent increasing stack size by introducing constants over 3.
//         // It prevents expression search space explosion like function_call(1,2,3,4,5,6, ..., 99).
//         if self.limit_introducing_constants_too_much && (state.stack.len() < 5) {
//             for &i in &self.available_integers {
//                 actions.push(Action::ShiftInteger(i));
//             }
//             for &f in &self.available_floats {
//                 actions.push(Action::ShiftFloat(f));
//             }
//             for s in &self.available_strings {
//                 actions.push(Action::ShiftString(s.clone()));
//             }
//         }

//         for sig in available_funcs {
//             let name = &sig.name;
//             let param_types = &sig.inputs;
//             if param_types.is_empty() {
//                 // Zero-ary operator - can always apply
//                 actions.push(Action::Reduce(name.clone()));
//             } else if state.stack.len() >= param_types.len() {
//                 let stack_suffix = &state.stack[state.stack.len() - param_types.len()..];
//                 if stack_suffix == param_types.as_slice() {
//                     actions.push(Action::Reduce(name.clone()));
//                 }
//             }
//         }

//         actions
//     }

//     pub fn step(
//         &self,
//         state: &SearchState,
//         action: Action,
//         available_funcs: &[runtime::ast::OperatorSignature],
//     ) -> Result<SearchState, String> {
//         let mut new_state = state.clone();

//         match action {
//             Action::Shift => {
//                 if let Some(param) = new_state.unprocessed_params.pop() {
//                     new_state.stack.push(param);
//                     new_state.sequence.push("!shift".to_string());
//                     Ok(new_state)
//                 } else {
//                     Err("Cannot shift: no unprocessed parameters remaining.".to_string())
//                 }
//             }
//             Action::ShiftInteger(val) => {
//                 // Technically it maps to Int or Float. We push Float per runtime/stdlib types.
//                 new_state.stack.push(TypeDecl::Float);
//                 new_state.sequence.push(val.to_string());
//                 Ok(new_state)
//             }
//             Action::ShiftFloat(val) => {
//                 new_state.stack.push(TypeDecl::Float);
//                 new_state.sequence.push(val.to_string());
//                 Ok(new_state)
//             }
//             Action::ShiftString(val) => {
//                 new_state.stack.push(TypeDecl::String);
//                 new_state.sequence.push(format!("\"{}\"", val));
//                 Ok(new_state)
//             }
//             Action::Reduce(func_name) => {
//                 let func_def = available_funcs
//                     .iter()
//                     .find(|sig| sig.name == *func_name)
//                     .ok_or_else(|| format!("Unknown function: {}", func_name))?;

//                 let param_types = &func_def.inputs;
//                 let ret_type = &func_def.output;

//                 if new_state.stack.len() < param_types.len() {
//                     return Err(format!("Cannot reduce {}: stack too small.", func_name));
//                 }

//                 // Check types and pop
//                 for expected_type in param_types.iter().rev() {
//                     let actual = new_state.stack.pop().unwrap();
//                     if &actual != expected_type {
//                         return Err(format!(
//                             "Type mismatch reducing {}: expected {:?}, got {:?}",
//                             func_name, expected_type, actual
//                         ));
//                     }
//                 }

//                 // Only push if not Void
//                 if *ret_type != TypeDecl::Void {
//                     new_state.stack.push(ret_type.clone());
//                 }

//                 // Add to sequence prefix explicitly for tracking (simulated postfix order for now,
//                 // but real serialization will build AST and convert using `dag_to_sequence`).
//                 new_state.sequence.push(func_name.clone());

//                 Ok(new_state)
//             }
//             Action::Done => {
//                 if new_state.unprocessed_params.is_empty()
//                     && new_state.stack.len() == 1
//                     && new_state.stack[0] == self.target_return
//                 {
//                     Ok(new_state)
//                 } else {
//                     Err("Exit condition not met.".to_string())
//                 }
//             }
//         }
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_behavior_search_logic() {
//         // From docs/behavior_search.md example
//         // Input params: [DataFrame, DataFrame, Float, DataFrame] (reversed since we pop from end: pop D first, then F, etc. Wait!
//         // Actually, the example in markdown says:
//         // Input: [D, D, F, D]. Increase order -> [D, D, F], [D]
//         // This means it pops the LAST element.
//         let mut state = SearchState {
//             unprocessed_params: vec![
//                 TypeDecl::DataFrame,
//                 TypeDecl::DataFrame,
//                 TypeDecl::Float,
//                 TypeDecl::DataFrame,
//             ],
//             stack: vec![],
//             sequence: vec![],
//         };

//         let env = SearchEnv::new(TypeDecl::DataFrame, vec![], vec![], vec![], true); // Target return: DataFrame

//         let available_funcs = vec![
//             runtime::ast::OperatorSpec {
//                 name: "consume_float".to_string(),
//                 inputs: vec![TypeDecl::Float],
//                 output: TypeDecl::Void,
//             },
//             runtime::ast::OperatorSpec {
//                 name: "cs_rank".to_string(),
//                 inputs: vec![TypeDecl::DataFrame],
//                 output: TypeDecl::DataFrame,
//             },
//             runtime::ast::OperatorSpec {
//                 name: "ts_diff".to_string(),
//                 inputs: vec![TypeDecl::DataFrame, TypeDecl::Float],
//                 output: TypeDecl::DataFrame,
//             },
//             runtime::ast::OperatorSpec {
//                 name: "divide".to_string(),
//                 inputs: vec![TypeDecl::DataFrame, TypeDecl::DataFrame],
//                 output: TypeDecl::DataFrame,
//             },
//         ];

//         // 1st step: Shift
//         state = env.step(&state, Action::Shift, &available_funcs).unwrap();
//         assert_eq!(
//             state.unprocessed_params,
//             vec![TypeDecl::DataFrame, TypeDecl::DataFrame, TypeDecl::Float]
//         );
//         assert_eq!(state.stack, vec![TypeDecl::DataFrame]);

//         // 2nd step: cs_rank
//         state = env
//             .step(
//                 &state,
//                 Action::Reduce("cs_rank".to_string()),
//                 &available_funcs,
//             )
//             .unwrap();
//         assert_eq!(
//             state.unprocessed_params,
//             vec![TypeDecl::DataFrame, TypeDecl::DataFrame, TypeDecl::Float]
//         );
//         assert_eq!(state.stack, vec![TypeDecl::DataFrame]);

//         // 3rd step: Shift
//         state = env.step(&state, Action::Shift, &available_funcs).unwrap();
//         assert_eq!(
//             state.unprocessed_params,
//             vec![TypeDecl::DataFrame, TypeDecl::DataFrame]
//         );
//         assert_eq!(state.stack, vec![TypeDecl::DataFrame, TypeDecl::Float]);

//         // Validate possible actions BEFORE 4th shift
//         let actions_before_4th = env.get_valid_actions(&state, &available_funcs);
//         assert!(actions_before_4th.contains(&Action::Shift));
//         assert!(actions_before_4th.contains(&Action::Reduce("ts_diff".to_string())));

//         // 4th step: Shift
//         state = env.step(&state, Action::Shift, &available_funcs).unwrap();
//         assert_eq!(state.unprocessed_params, vec![TypeDecl::DataFrame]);
//         assert_eq!(
//             state.stack,
//             vec![TypeDecl::DataFrame, TypeDecl::Float, TypeDecl::DataFrame]
//         );

//         // Validate possible actions AFTER 4th shift
//         let actions = env.get_valid_actions(&state, &available_funcs);
//         assert!(actions.contains(&Action::Shift));
//         assert!(actions.contains(&Action::Reduce("cs_rank".to_string())));
//     }

//     #[test]
//     fn test_consume_minimal_integration() {
//         use parser::parser::parse;
//         use parser::program::{BehaviorDecl, Declaration};

//         let src = std::fs::read_to_string("../examples/behavior_1.cm").expect("Read failed");
//         let program = parse(&src).expect("Failed to parse behavior_1.cm");

//         let available_funcs = runtime::ast::OperatorSpec::get_available_funcs();

//         let mut target_behavior = None;
//         for decl in &program.declarations {
//             if let Declaration::Behavior(b) = decl {
//                 if b.name == "Comparator" {
//                     target_behavior = Some(b.clone());
//                 }
//             }
//         }

//         let behavior = target_behavior.expect("Failed to find Comparator behavior");

//         println!("Inferring action candidates from the code:");
//         for f in &available_funcs {
//             println!("- {}({:?}) -> {:?}", f.name, f.inputs, f.output);
//         }

//         println!("\nGenerating sample expression trees and evaluating using runtime...");
//         let mut runtime = Runtime::new(100, "../data");
//         let samples =
//             generate_top_k_samples(&behavior, &available_funcs, 3, |_| true, &mut runtime);

//         for (i, sample) in samples.iter().enumerate() {
//             println!("Sample {}: Fitness = {:?}", i + 1, sample.fitness);
//             println!("  Actions: {:?}", sample.actions);
//         }

//         assert!(
//             !samples.is_empty(),
//             "Should generate at least one valid expression tree"
//         );
//     }
// }
