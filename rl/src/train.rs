use crate::action::Action;
use crate::env::Environment;
use crate::model::Model;
use crate::trajectory::Trajectory;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

pub struct BatchConfig {
    pub batch_size: usize,
    pub trajectories: Vec<Trajectory>,
}

fn encode_state_dummy<B: Backend>(device: &B::Device) -> Tensor<B, 3, Int> {
    // RNN expects [batch_size, seq_length, 2]
    // where the 2 features are parent and sibling action indices.
    let state_data = TensorData::from([[[0, 0]]]);
    Tensor::<B, 3, Int>::from_data(state_data.convert::<i32>(), device)
}

impl<'a> Environment<'a> {
    pub fn sample_trajectory<B: Backend>(
        &mut self,
        model: &Model<B>,
        device: &B::Device,
    ) -> Trajectory {
        self.reset();
        let mut trajectory: Trajectory = Vec::new();
        let action_space = self.action_space.clone();

        for _ in 0..self.max_length {
            let valid_actions: Vec<Action> = self.state.get_valid_actions(&action_space);

            if valid_actions.is_empty() {
                break;
            }

            let mut available_mask = vec![false; action_space.size()];
            for a in &valid_actions {
                available_mask[action_space.get_idx(a)] = true;
            }

            let available_tensor = burn::tensor::Tensor::<B, 3, burn::tensor::Bool>::from_bool(
                burn::tensor::TensorData::new(available_mask, [1, 1, action_space.size()]),
                device,
            );

            let state_tensor = encode_state_dummy::<B>(device);

            // Inference
            let logits: Tensor<B, 3> = model.forward(state_tensor, available_tensor);

            let probs = burn::tensor::activation::softmax(logits, 2);
            let tensor_data = probs.into_data();
            let probs_data: &[f32] = tensor_data.as_slice().expect("Failed to get f32 slice");

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

            let dist = WeightedIndex::new(&valid_probs)
                .unwrap_or_else(|_| WeightedIndex::new(vec![1.0; valid_actions.len()]).unwrap());
            let mut rng = thread_rng();
            let action = valid_actions[dist.sample(&mut rng)].clone();

            let item = self.step(action.clone());
            trajectory.push(item);

            if matches!(action, Action::Done) {
                break;
            }
        }
        trajectory
    }

    pub fn sample_trajectories<B: Backend>(&mut self, model: &Model<B>, device: &B::Device) {
        self.config.trajectories.clear();
        for _ in 0..self.config.batch_size {
            let traj = self.sample_trajectory(model, device);
            self.config.trajectories.push(traj);
        }
    }

    pub fn calculate_fitness(&mut self) -> Vec<f64> {
        let mut fitnesses = Vec::new();
        for traj in &self.config.trajectories {
            let mut total_reward = 0.0;
            for item in traj {
                total_reward += item.reward;
            }
            fitnesses.push(total_reward);
        }
        fitnesses
    }

    pub fn calculate_gradient<B: AutodiffBackend>(
        &self,
        model: &Model<B>,
        device: &B::Device,
    ) -> GradientsParams {
        let mut total_loss = Tensor::<B, 1>::zeros([1], device);

        for traj in &self.config.trajectories {
            let mut traj_loss = Tensor::<B, 1>::zeros([1], device);
            let mut reward_sum = 0.0;
            for item in traj {
                reward_sum += item.reward;
            }

            // Dummy loss calculation since exact REINFORCE needs re-evaluation or stored log_probs
            let dummy_loss = Tensor::<B, 1>::zeros([1], device);
            traj_loss = traj_loss + dummy_loss * reward_sum;
            total_loss = total_loss + traj_loss;
        }

        let gradients = total_loss.backward();
        GradientsParams::from_grads(gradients, model)
    }

    pub fn update_weight<B: AutodiffBackend, O: Optimizer<Model<B>, B>>(
        optimizer: &mut O,
        model: Model<B>,
        gradients: GradientsParams,
        lr: f64,
    ) -> Model<B> {
        optimizer.step(lr, model, gradients)
    }
}
