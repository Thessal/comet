// use crate::action::{Action, SearchState};
// use crate::model::TransformerModel;
// use burn::module::{AutodiffModule, Module};
// use burn::optim::{GradientsParams, Optimizer};
// use burn::tensor::backend::{AutodiffBackend, Backend};
// use burn::tensor::{Int, Tensor, TensorData};
// use parser::behavior::BehaviorDecl;
// use parser::expr::Ident;
// use std::collections::HashMap;
// use stdlib::types::Signal;

// /// Helper function to perform one explicit gradient step
// pub fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
//     loss: Tensor<B, 1>,
//     module: M,
//     optimizer: &mut impl Optimizer<M, B>,
//     learning_rate: f64,
// ) -> M {
//     let gradients = loss.backward();
//     let gradient_params = GradientsParams::from_grads(gradients, &module);
//     optimizer.step(learning_rate, module, gradient_params)
// }

// fn sample_trajectory<B: Backend>(
//     inference_model: &TransformerModel<B>,
//     behavior: &BehaviorDecl,
//     available_funcs: &[runtime::ast::OperatorSpec],
//     device: &B::Device,
//     temperature: f64,
// ) -> Trajectory {
//     let env = SearchEnv::new(
//         behavior.return_type.clone(),
//         behavior.integers.clone().unwrap_or_default(),
//         behavior.floats.clone().unwrap_or_default(),
//         behavior.strings.clone().unwrap_or_default(),
//         true,
//     );

//     let mut state = SearchState {
//         params: behavior
//             .args
//             .iter()
//             .rev()
//             .map(|arg| arg.type_decl.clone())
//             .collect(),
//         stack: vec![],
//         // sequence: vec![],
//     };

//     let action_space = crate::action::ActionSpace::new(behavior, available_funcs);
//     let action_vocab_size = action_space.size();

//     let mut traj = Trajectory {
//         states: vec![],
//         actions: vec![],
//         valid_actions_mask: vec![],
//         sequence: vec![],
//     };

//     for _ in 0..100 {
//         let valid_actions = env.get_valid_actions(&state, available_funcs);
//         if valid_actions.is_empty() {
//             break;
//         }
//         if valid_actions.contains(&Action::Done) {
//             let encoded = crate::supervised::encode_state(
//                 &state,
//                 state
//                     .sequence
//                     .last()
//                     .map_or(0, |a| action_space.action_to_id(&Action::from_string(a))),
//             );
//             traj.states.push(encoded);
//             let action_id = action_space.action_to_id(&Action::Done);
//             traj.actions.push(action_id);

//             let mut available_mask = vec![false; action_vocab_size];
//             for a in &valid_actions {
//                 let id = action_space.action_to_id(a);
//                 if id < action_vocab_size {
//                     available_mask[id] = true;
//                 }
//             }
//             traj.valid_actions_mask.push(available_mask);

//             break; // Stop immediately upon reaching Done state natively.
//         }

//         let prev_action_id = state
//             .sequence
//             .last()
//             .map_or(0, |a| action_space.action_to_id(&Action::from_string(a)));

//         let encoded = crate::supervised::encode_state(&state, prev_action_id);
//         traj.states.push(encoded);

//         // Forward pass
//         let encoded_i32: Vec<i32> = encoded.iter().map(|&x| x as i32).collect();
//         let input_tensor =
//             Tensor::<B, 3, Int>::from_data(TensorData::new(encoded_i32, [1, 1, 8]), device);

//         let mut available_mask = vec![false; action_vocab_size];
//         for a in &valid_actions {
//             let id = action_space.action_to_id(a);
//             if id < action_vocab_size {
//                 available_mask[id] = true;
//             }
//         }
//         traj.valid_actions_mask.push(available_mask.clone());

//         let available_tensor = burn::tensor::Tensor::<B, 3, burn::tensor::Bool>::from_bool(
//             TensorData::new(available_mask, [1, 1, action_vocab_size]),
//             device,
//         );

//         let logits = inference_model.forward(input_tensor, available_tensor);
//         let probs = burn::tensor::activation::softmax(logits / temperature, 2);
//         let probs_data = probs.into_data().to_vec::<f32>().unwrap();

//         let mut valid_probs: Vec<f32> = valid_actions
//             .iter()
//             .map(|a| {
//                 let id = action_space.action_to_id(a);
//                 probs_data.get(id).copied().unwrap_or(0.0)
//             })
//             .collect();

//         let sum: f32 = valid_probs.iter().sum();
//         if sum > 1e-6 {
//             for p in &mut valid_probs {
//                 *p /= sum;
//             }
//         } else {
//             let uniform = 1.0 / valid_actions.len() as f32;
//             for p in &mut valid_probs {
//                 *p = uniform;
//             }
//         }

//         use rand::distributions::WeightedIndex;
//         use rand::prelude::*;
//         let dist = WeightedIndex::new(&valid_probs).unwrap();
//         let mut rng = thread_rng();
//         let chosen_action = valid_actions[dist.sample(&mut rng)].clone();

//         let action_id = action_space.action_to_id(&chosen_action);
//         traj.actions.push(action_id);

//         if chosen_action == Action::Done {
//             break;
//         }

//         state = env.step(&state, chosen_action, available_funcs).unwrap();
//     }
//     traj.sequence = state.sequence;
//     traj
// }

// pub fn train_rl<B: AutodiffBackend>(
//     mut model: TransformerModel<B>,
//     behavior: &BehaviorDecl,
//     available_funcs: &[runtime::ast::OperatorSpec],
//     runtime: &mut runtime::runtime::Runtime,
//     call_args: Vec<String>,
//     epochs: usize,
//     batch_size: usize,
//     learning_rate: f64,
//     lambda_complexity: f64,
//     entropy_weight: f64,
// ) -> TransformerModel<B> {
//     use burn::optim::AdamConfig;
//     let device = <B as Backend>::Device::default();
//     let config_optim = AdamConfig::new();
//     let mut optimizer = config_optim.init();

//     println!("--- Launching Target Reinforcement Learning Training ---");

//     for epoch in 0..epochs {
//         println!("RL Epoch {}/{}", epoch + 1, epochs);
//         let inference_model = model.valid();

//         let mut trajs = Vec::new();
//         for _ in 0..batch_size {
//             trajs.push(sample_trajectory(
//                 &inference_model,
//                 behavior,
//                 available_funcs,
//                 &device,
//                 1.0,
//             ));
//         }

//         let sequences: Vec<&[String]> = trajs.iter().map(|t| t.sequence.as_slice()).collect();

//         let mut parsed_outputs = Vec::new();
//         for seq in &sequences {
//             match runtime.evaluate_sequence(seq, call_args.clone()) {
//                 Ok(stdlib::Signal::DataFrame(output)) => {
//                     parsed_outputs.push(Some(output));
//                 }
//                 _ => {
//                     parsed_outputs.push(None); // Penality placeholder
//                 }
//             }
//         }

//         let mut valid_refs = Vec::new();
//         for out in &parsed_outputs {
//             if let Some(o) = out {
//                 valid_refs.push(o.as_slice());
//             } else {
//                 valid_refs.push(&[]);
//             }
//         }

//         let batch_fitness =
//             runtime::stats::evaluate_fitness_batch_add_value(&mut runtime.dmgr, &valid_refs);

//         let fitnesses: Vec<f64> = batch_fitness
//             .into_iter()
//             .map(|metrics| runtime::stats::fitness_summary(&metrics))
//             .collect();

//         let mut rewards = Vec::new();
//         for (i, traj) in trajs.iter().enumerate() {
//             let reward = fitnesses[i] - lambda_complexity * (traj.sequence.len() as f64);
//             rewards.push(reward);
//         }

//         let baseline: f64 = rewards.iter().sum::<f64>() / (batch_size as f64);
//         let advantages: Vec<f64> = rewards.iter().map(|r| r - baseline).collect();

//         // Print some intermediate info
//         println!(
//             "  Avg Reward: {:.4} | Baseline: {:.4} | Example Seq Len: {}",
//             rewards.iter().sum::<f64>() / (batch_size as f64),
//             baseline,
//             trajs.get(0).map(|t| t.sequence.len()).unwrap_or(0)
//         );

//         let max_seq_len = trajs
//             .iter()
//             .map(|t| t.states.len())
//             .max()
//             .unwrap_or(1)
//             .max(1);

//         let action_space = crate::action::ActionSpace::new(behavior, available_funcs);
//         let action_vocab_size = action_space.size();

//         let mut inputs_data = Vec::with_capacity(batch_size * max_seq_len * 8);
//         let mut actions_data = Vec::with_capacity(batch_size * max_seq_len);
//         let mut advantages_data = Vec::with_capacity(batch_size * max_seq_len);
//         let mut mask_data = Vec::with_capacity(batch_size * max_seq_len);
//         let mut valid_mask_data = Vec::with_capacity(batch_size * max_seq_len * action_vocab_size);

//         for (i, traj) in trajs.iter().enumerate() {
//             let seq_len = traj.states.len();
//             let adv = advantages[i] as f32;
//             for t in 0..max_seq_len {
//                 if t < seq_len {
//                     inputs_data.extend_from_slice(&traj.states[t]);
//                     actions_data.push(traj.actions[t] as i32);
//                     advantages_data.push(adv);
//                     mask_data.push(1.0f32);
//                     valid_mask_data.extend_from_slice(&traj.valid_actions_mask[t]);
//                 } else {
//                     inputs_data.extend_from_slice(&[0; 8]);
//                     actions_data.push(0);
//                     advantages_data.push(0.0);
//                     mask_data.push(0.0);
//                     let mut pad_mask = vec![false; action_vocab_size];
//                     pad_mask[0] = true;
//                     valid_mask_data.extend(pad_mask);
//                 }
//             }
//         }

//         let inputs_tensor = Tensor::<B, 3, Int>::from_data(
//             TensorData::new(
//                 inputs_data
//                     .into_iter()
//                     .map(|v| v as i32)
//                     .collect::<Vec<_>>(),
//                 [batch_size, max_seq_len, 8],
//             ),
//             &device,
//         );

//         let action_tensor = Tensor::<B, 3, Int>::from_data(
//             TensorData::new(actions_data, [batch_size, max_seq_len, 1]),
//             &device,
//         );

//         let advantages_tensor = Tensor::<B, 2>::from_data(
//             TensorData::new(advantages_data, [batch_size, max_seq_len]),
//             &device,
//         );

//         let mask_tensor = Tensor::<B, 2>::from_data(
//             TensorData::new(mask_data, [batch_size, max_seq_len]),
//             &device,
//         );

//         let available_actions_tensor = Tensor::<B, 3, burn::tensor::Bool>::from_bool(
//             TensorData::new(
//                 valid_mask_data,
//                 [batch_size, max_seq_len, action_vocab_size],
//             ),
//             &device,
//         );

//         // Forward Pass (WITH gradients across batch!)
//         let logits = model.forward(inputs_tensor, available_actions_tensor.clone()); // [batch, max_seq_len, vocab_size]
//         let probs = burn::tensor::activation::softmax(logits.clone(), 2);
//         let log_probs = burn::tensor::activation::log_softmax(logits, 2);

//         // Select the log prob of the action taken
//         let selected_log_probs = log_probs
//             .clone()
//             .gather(2, action_tensor)
//             .reshape([batch_size, max_seq_len]);

//         // Policy Loss = -1 * log_prob * advantage
//         let policy_loss =
//             (selected_log_probs * advantages_tensor * mask_tensor.clone()).mul_scalar(-1.0_f32);
//         let policy_loss_mean = policy_loss.sum_dim(1).mean();

//         // Entropy Bonus = - sum( p * log(p) )
//         // To avoid NaN from 0 * -inf, we mask log_probs to 0 where available_actions is false
//         let is_invalid = available_actions_tensor.bool_not();
//         let safe_log_probs = log_probs.mask_fill(is_invalid, 0.0);
//         let entropy = (probs * safe_log_probs)
//             .sum_dim(2)
//             .mul_scalar(-1.0_f32)
//             .reshape([batch_size, max_seq_len]);
//         let entropy_mean = (entropy * mask_tensor).sum_dim(1).mean();

//         // Total Loss to minimize
//         let total_loss = policy_loss_mean - entropy_mean.mul_scalar(entropy_weight as f32);

//         // Perform gradient update step
//         model = update_parameters(
//             total_loss.reshape([1]),
//             model,
//             &mut optimizer,
//             learning_rate,
//         );
//     }

//     println!("--- RL Training Completed ---");
//     model
// }
