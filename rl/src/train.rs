use crate::env::Environment;
use crate::loss;
use crate::model::Model;
use crate::state::SearchState;
use crate::trajectory::{self, Trajectory};
use crate::{action::Action, trajectory::Step};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use tch::{
    Device, Kind,
    Kind::Float,
    Tensor,
    nn::{self, Optimizer, OptimizerConfig},
};

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
            let _state = self.state_embed(&self.state, device);
            let observation = _state;

            let logits_not_masked = match model {
                Model::RnnModel(rnnmodel) => {
                    tch::no_grad(|| observation.unsqueeze(0).apply(rnnmodel))
                }
                _ => {
                    panic!()
                }
            };
            // Mask invalid actions
            let valid_actions: Vec<Action> = self.state.get_valid_actions(&action_space);
            let available_actions: Tensor = action_space.calculate_mask(&valid_actions);
            let is_invalid = available_actions.logical_not();
            let logits = logits_not_masked.masked_fill(&is_invalid, f64::NEG_INFINITY);

            let sampled_actions: Vec<i64> =
                tch::no_grad(|| logits.softmax(1, Float).multinomial(1, true))
                    .try_into()
                    .unwrap();
            let action_idx = sampled_actions[0];
            let action: Action = action_space.get_action(action_idx as usize);
            let step = self.step(action);
            let is_done = step.is_done();
            trajectory.push(step);
            if is_done {
                break;
            }
        }
        self.reset();
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

    /// Trains an agent using the policy gradient algorithm.
    pub fn run(&mut self, model: &Model, device: Device) {
        let vs = nn::VarStore::new(device);
        let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();
        println!("action space: {:?}", self.action_space);
        for epoch_idx in 0..50 {
            self.sample_trajectories(model, device);
            let trajectories = &self.config.trajectories;

            let sum_r: f64 = trajectories
                .iter()
                .map(|traj| traj.iter().map(|s| s.reward).sum::<f64>())
                .sum();
            let episodes = trajectories.len() as i64; // counts even if is_done is false.. is it okay?
            println!(
                "epoch: {:<3} episodes: {:<5} avg reward per episode: {:.2}",
                epoch_idx,
                episodes,
                sum_r / episodes as f64
            );

            // Train the model via policy gradient on the rollout data.
            // Flattening the result, for standard REINFORCE on non-episodic tasks.
            let _batch_size: i64 = trajectories.iter().map(|traj| traj.len() as i64).sum();
            let rewards: Vec<Vec<f64>> = loss::policy_gradient::accumulate_rewards(&trajectories);
            let traj_flat: Trajectory = trajectories.clone().into_iter().flatten().collect();
            let reward_flat: Vec<f64> = rewards.into_iter().flatten().collect();
            let actions_flat: Vec<Action> =
                traj_flat.iter().map(|step| step.action.clone()).collect();
            let states_flat: Vec<SearchState> =
                traj_flat.iter().map(|step| step.state.clone()).collect();
            let reward_flat = Tensor::from_slice(&reward_flat).to_kind(Float);
            let actions_flat: Vec<i64> = actions_flat
                .iter()
                .map(|action| self.action_space.get_idx(action) as i64)
                .collect();
            let observations_flat: Vec<Tensor> = states_flat
                .iter()
                .map(|state| self.state_embed(state, device))
                .collect();

            todo!();
            // let action_mask = Tensor::zeros([batch_size, 2], tch::kind::FLOAT_CPU)
            //     .scatter_value(1, &actions, 1.0); // FIXME
            // let obs: Vec<Tensor> = traj_flat.into_iter().map(|s| s.obs).collect();
            let logits_not_masked = Tensor::stack(&observations_flat, 0).apply(model);
            // let action_mask = &available_actions.logical_not()
            // let log_probs =
            //     (action_mask * logits.log_softmax(1, Float)).sum_dim_intlist(1, false, Float);
            let log_probs = // without mask
                (logits_not_masked.log_softmax(1, Float)).sum_dim_intlist(1, false, Float);
            let loss = -(reward_flat * log_probs).mean(Float);
            opt.backward_step(&loss)
        }
    }
}
