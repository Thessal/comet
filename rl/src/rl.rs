use crate::model::TransformerModel;
use crate::search::{Action, SearchEnv, SearchState};
use burn::module::{AutodiffModule, Module};
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use parser::program::{BehaviorDecl, Ident, TypeDecl};
use std::collections::HashMap;

/// Helper function to perform one explicit gradient step
pub fn update_parameters<B: AutodiffBackend, M: AutodiffModule<B>>(
    loss: Tensor<B, 1>,
    module: M,
    optimizer: &mut impl Optimizer<M, B>,
    learning_rate: f64,
) -> M {
    let gradients = loss.backward();
    let gradient_params = GradientsParams::from_grads(gradients, &module);
    optimizer.step(learning_rate, module, gradient_params)
}

struct Trajectory {
    states: Vec<[usize; 8]>,
    actions: Vec<usize>,
    sequence: Vec<String>,
}

fn sample_trajectory<B: Backend>(
    inference_model: &TransformerModel<B>,
    behavior: &BehaviorDecl,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    device: &B::Device,
    temperature: f64,
) -> Trajectory {
    let env = SearchEnv::new(
        behavior.return_type.clone(),
        behavior.integers.clone().unwrap_or_default(),
        behavior.floats.clone().unwrap_or_default(),
        behavior.strings.clone().unwrap_or_default(),
        true,
    );

    let mut state = SearchState {
        unprocessed_params: behavior
            .args
            .iter()
            .rev()
            .map(|arg| arg.type_decl.clone())
            .collect(),
        stack: vec![],
        sequence: vec![],
    };

    let mut traj = Trajectory {
        states: vec![],
        actions: vec![],
        sequence: vec![],
    };

    for _ in 0..100 {
        let valid_actions = env.get_valid_actions(&state, available_funcs);
        if valid_actions.is_empty() {
            break;
        }
        if valid_actions.contains(&Action::Done) {
            let encoded = crate::supervised::encode_state(
                &state,
                state.sequence.last().map_or(0, |a| {
                    crate::supervised::action_to_id(
                        &crate::supervised::string_to_action(a),
                        behavior,
                        available_funcs,
                    )
                }),
            );
            traj.states.push(encoded);
            let action_id = crate::supervised::action_to_id(&Action::Done, behavior, available_funcs);
            traj.actions.push(action_id);
            break; // Stop immediately upon reaching Done state natively.
        }

        let prev_action_id = state.sequence.last().map_or(0, |a| {
            crate::supervised::action_to_id(
                &crate::supervised::string_to_action(a),
                behavior,
                available_funcs,
            )
        });

        let encoded = crate::supervised::encode_state(&state, prev_action_id);
        traj.states.push(encoded);

        // Forward pass
        let encoded_i32: Vec<i32> = encoded.iter().map(|&x| x as i32).collect();
        let input_tensor = Tensor::<B, 3, Int>::from_data(
            TensorData::new(encoded_i32, [1, 1, 8]),
            device,
        );
        let logits = inference_model.forward(input_tensor);
        let probs = burn::tensor::activation::softmax(logits / temperature, 2);
        let probs_data = probs.into_data().to_vec::<f32>().unwrap();

        let mut valid_probs: Vec<f32> = valid_actions
            .iter()
            .map(|a| {
                let id = crate::supervised::action_to_id(a, behavior, available_funcs);
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

        use rand::distributions::WeightedIndex;
        use rand::prelude::*;
        let dist = WeightedIndex::new(&valid_probs).unwrap();
        let mut rng = thread_rng();
        let chosen_action = valid_actions[dist.sample(&mut rng)].clone();

        let action_id = crate::supervised::action_to_id(&chosen_action, behavior, available_funcs);
        traj.actions.push(action_id);

        if chosen_action == Action::Done {
            break;
        }

        state = env.step(&state, chosen_action, available_funcs).unwrap();
    }
    traj.sequence = state.sequence;
    traj
}

pub fn train_rl<B: AutodiffBackend, F>(
    mut model: TransformerModel<B>,
    behavior: &BehaviorDecl,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    mut eval_fn: F,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    lambda_complexity: f64,
    entropy_weight: f64,
) -> TransformerModel<B>
where
    F: FnMut(&[String]) -> f64,
{
    use burn::optim::AdamConfig;
    let device = <B as Backend>::Device::default();
    let config_optim = AdamConfig::new();
    let mut optimizer = config_optim.init();

    println!("--- Launching Target Reinforcement Learning Training ---");

    for epoch in 0..epochs {
        println!("RL Epoch {}/{}", epoch + 1, epochs);
        let inference_model = model.valid();

        let mut trajs = Vec::new();
        for _ in 0..batch_size {
            trajs.push(sample_trajectory(&inference_model, behavior, available_funcs, &device, 1.0));
        }

        let mut rewards = Vec::new();
        for traj in &trajs {
            let fitness = eval_fn(&traj.sequence);
            let reward = fitness - lambda_complexity * (traj.sequence.len() as f64);
            rewards.push(reward);
        }

        let baseline: f64 = rewards.iter().sum::<f64>() / (batch_size as f64);
        let advantages: Vec<f64> = rewards.iter().map(|r| r - baseline).collect();

        // Print some intermediate info
        println!(
            "  Avg Reward: {:.4} | Baseline: {:.4} | Example Seq Len: {}",
            rewards.iter().sum::<f64>() / (batch_size as f64),
            baseline,
            trajs.get(0).map(|t| t.sequence.len()).unwrap_or(0)
        );

        let max_seq_len = trajs.iter().map(|t| t.states.len()).max().unwrap_or(1).max(1);

        let mut inputs_data = Vec::with_capacity(batch_size * max_seq_len * 8);
        let mut actions_data = Vec::with_capacity(batch_size * max_seq_len);
        let mut advantages_data = Vec::with_capacity(batch_size * max_seq_len);
        let mut mask_data = Vec::with_capacity(batch_size * max_seq_len);

        for (i, traj) in trajs.iter().enumerate() {
            let seq_len = traj.states.len();
            let adv = advantages[i] as f32;
            for t in 0..max_seq_len {
                if t < seq_len {
                    inputs_data.extend_from_slice(&traj.states[t]);
                    actions_data.push(traj.actions[t] as i32);
                    advantages_data.push(adv);
                    mask_data.push(1.0f32);
                } else {
                    inputs_data.extend_from_slice(&[0; 8]);
                    actions_data.push(0);
                    advantages_data.push(0.0);
                    mask_data.push(0.0);
                }
            }
        }

        let inputs_tensor = Tensor::<B, 3, Int>::from_data(
            TensorData::new(inputs_data.into_iter().map(|v| v as i32).collect::<Vec<_>>(), [batch_size, max_seq_len, 8]),
            &device,
        );

        let action_tensor = Tensor::<B, 3, Int>::from_data(
            TensorData::new(actions_data, [batch_size, max_seq_len, 1]),
            &device,
        );

        let advantages_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(advantages_data, [batch_size, max_seq_len]),
            &device,
        );

        let mask_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, [batch_size, max_seq_len]),
            &device,
        );

        // Forward Pass (WITH gradients across batch!)
        let logits = model.forward(inputs_tensor); // [batch, max_seq_len, vocab_size]
        let probs = burn::tensor::activation::softmax(logits.clone(), 2);
        let log_probs = burn::tensor::activation::log_softmax(logits, 2);

        // Select the log prob of the action taken
        let selected_log_probs = log_probs.clone().gather(2, action_tensor).reshape([batch_size, max_seq_len]);

        // Policy Loss = -1 * log_prob * advantage
        let policy_loss = (selected_log_probs * advantages_tensor * mask_tensor.clone()).mul_scalar(-1.0_f32);
        let policy_loss_mean = policy_loss.sum_dim(1).mean();

        // Entropy Bonus = - sum( p * log(p) ) 
        let entropy = (probs * log_probs).sum_dim(2).mul_scalar(-1.0_f32).reshape([batch_size, max_seq_len]);
        let entropy_mean = (entropy * mask_tensor).sum_dim(1).mean();

        // Total Loss to minimize
        let total_loss = policy_loss_mean - entropy_mean.mul_scalar(entropy_weight as f32);

        // Perform gradient update step
        model = update_parameters(total_loss.reshape([1]), model, &mut optimizer, learning_rate);
    }
    
    println!("--- RL Training Completed ---");
    model
}
