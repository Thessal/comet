use crate::action::Action;
use crate::env::Environment;
use crate::model::TransformerModel;
use crate::state::SearchState;
use crate::trajectory::Trajectory;
use crate::trajectory::TrajectoryItem;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss;
use burn::optim::AdamConfig;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use parser::behavior::BehaviorDecl;
use parser::expr::Ident;
use runtime::ast::PolishExpr;
use runtime::ast::Tree;
use runtime::pnl::PnlCalculator;
use runtime::pnl::PnlResult;
use runtime::stats::Stats;
use std::collections::HashMap;
use stdlib::types::Signal;

pub struct BatchConfig<B: Backend> {
    pub model: TransformerModel<B>,
    pub optimizer: AdamConfig,
    pub trajectories: Vec<Trajectory>,
}

impl<'a, B: AutodiffBackend> Environment<'a, B> {
    fn sample_trajectory(
        &mut self,
        model: &TransformerModel<B>,
        config: &BatchConfig<B>,
    ) -> Trajectory {
        let mut trajectory: Trajectory = Vec::new();
        for _ in 0..self.max_length {
            let action_space = &self.action_space;
            let valid_actions: Vec<Action> = self.state.get_valid_actions(action_space);

            let mut available_mask = vec![false; action_space.size()];
            for a in &valid_actions {
                available_mask[action_space.get_idx(a)] = true;
            }
            // Use model's internal device (dummy fetch since we don't have direct device accessor here, we assume default or use a dummy)
            let device = B::Device::default();
            let available_tensor = burn::tensor::Tensor::<B, 3, burn::tensor::Bool>::from_bool(
                burn::tensor::TensorData::new(available_mask, [1, 1, action_space.size()]),
                &device,
            );

            let state_data = self.state.to_tensor_data();
            let state_tensor = Tensor::<B, 3, Int>::from_data(state_data, &device);

            //Inference
            let logits: Tensor<B, 3> = config.model.forward(state_tensor, available_tensor);

            let probs = burn::tensor::activation::softmax(logits, 2);
            let probs_data = probs.into_data().to_vec::<f32>().unwrap();

            let mut valid_probs: Vec<f32> = valid_actions
                .iter()
                .map(|a| {
                    let id = action_space.get_idx(a);
                    probs_data.get(id).copied().unwrap_or(0.0)
                })
                .collect();

            let sum: f32 = valid_probs.iter().sum();
            if sum > 1e-6 {
                for p in &mut valid_probs {
                    *p /= sum;
                }
            } else {
                let uniform = 1.0 / valid_actions.len() as f32;
                for p in &mut valid_probs {
                    *p = uniform;
                }
            }

            // Dummy action selection
            let action = valid_actions[0].clone();

            let (new_state, done) = self
                .state
                .clone()
                .apply_action(action.clone(), &mut self.runtime);
            self.state = new_state.clone();

            trajectory.push(TrajectoryItem {
                state: self.state.clone(),
                action,
                reward: 0.0,
                next_state: Some(new_state.clone()),
                sequence: self.state.expr.clone(),
            });

            if done {
                break;
            }
        }
        trajectory
    }

    pub fn sample_trajectories(
        &mut self,
        model: &TransformerModel<B>,
        config: &mut BatchConfig<B>,
    ) {
        for i in 0..config.trajectories.len() {
            config.trajectories[i] = self.sample_trajectory(model, config);
        }
    }

    pub fn calculate_fitness(&mut self, config: &BatchConfig<B>) -> Vec<f64> {
        let trajectories = &config.trajectories;
        let sequences: Vec<&PolishExpr> = trajectories
            .iter()
            .map(|trajectory| &trajectory.iter().last().unwrap().sequence)
            .collect();
        let asts: Vec<Tree> = sequences.into_iter().map(|s| s.into()).collect();
        let positions: Vec<Signal> = asts.iter().map(|ast| self.runtime.run(ast)).collect();
        let pnls: Vec<PnlResult> = positions
            .iter()
            .map(|pos| self.pnl_calc.pnl(&pos))
            .collect();
        let fitnesses: Vec<Stats> = pnls.iter().map(|pnl| pnl.into()).collect(); // reward at the end of trajectory.
        todo!("Think about how to add intermediate step rewards, not only reward of episode")
    }

    pub fn calculate_gradient(&self, config: &BatchConfig<B>) -> GradientsParams {
        let model = &config.model;
        // dummy return
        // let empty_loss = Tensor::<B, 1>::zeros([1], &B::Device::default());
        let loss: Tensor<B, 1> = todo!();
        let gradients = loss.backward();
        GradientsParams::from_grads(gradients, model)
    }

    pub fn update_weight(&mut self, gradients: GradientsParams) {
        let model = &self.config.model;
        todo!()
    }
}

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
