use crate::action::Action;
use crate::env::Environment;
use crate::model::Model;
use crate::trajectory::Trajectory;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use tch::{Device, Kind, Tensor, nn::Optimizer};

pub struct BatchConfig {
    pub batch_size: usize,
    pub trajectories: Vec<Trajectory>,
}

impl<'a> Environment<'a> {
    pub fn sample_trajectory(&mut self, model: &Model, device: Device) -> Trajectory {
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

            let available_tensor = Tensor::from_slice(&available_mask)
                .view([1, 1, action_space.size() as i64])
                .to_device(device);

            // TODO: implement state_embed properly using tch
            let state_tensor = Tensor::zeros([1, 1, 2], (Kind::Int64, device));

            // Inference
            let logits = model.forward(&state_tensor, &available_tensor);

            let probs = logits.softmax(-1, Kind::Float);
            let probs_data: Vec<f32> = probs.try_into().expect("Failed to get f32 slice");

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

    pub fn sample_trajectories(&mut self, model: &Model, device: Device) {
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

    pub fn calculate_gradient(&self, _model: &Model, device: Device) {
        let mut total_loss = Tensor::zeros([1], (Kind::Float, device));

        for traj in &self.config.trajectories {
            let mut traj_loss = Tensor::zeros([1], (Kind::Float, device));
            let reward_sum = 0.0;
            // for item in traj {
            //     reward_sum += item.reward;
            // }

            // Dummy loss calculation since exact REINFORCE needs re-evaluation or stored log_probs
            let _dummy_loss = Tensor::zeros([1], (Kind::Float, device));
            traj_loss = traj_loss + _dummy_loss * reward_sum;
            total_loss = total_loss + traj_loss;
            todo!("Implement REINFORCE");
        }

        total_loss.backward();
    }

    pub fn update_weight(optimizer: &mut Optimizer) {
        optimizer.step();
    }
}
