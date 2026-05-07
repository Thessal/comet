/// TODO: https://github.com/benbaarber/rl/blob/main/Cargo.toml
/// https://github.com/LaurentMazare/tch-rs/tree/main/examples/reinforcement-learning

pub(crate) mod policy_gradient {

    use tch::{Device, Kind::Float, Tensor};

    use crate::{
        action::Action, env::Environment, model::Model, state::SearchState, trajectory::Trajectory,
    };

    pub fn calc_intermediate_reward() -> f64 {
        -0.1
    }

    pub fn calc_terminal_reward(fitness: f64) -> f64 {
        fitness
    }

    // policy gradient
    pub fn accumulate_rewards(trajectories: &Vec<Trajectory>) -> Vec<Vec<f64>> {
        let mut rewards: Vec<Vec<f64>> = vec![];
        for traj in trajectories {
            let mut rewards_: Vec<f64> = vec![];
            let mut acc_reward = 0f64;
            for step in traj.iter().rev() {
                acc_reward += step.reward;
                rewards_.push(acc_reward);
            }
            rewards_.reverse();
            rewards.push(rewards_);
        }
        rewards
    }

    pub fn calculate_loss(env: &Environment, model: &Model, device: Device) -> Tensor {
        let trajectories = &env.config.trajectories;
        let sum_r: f64 = trajectories
            .iter()
            .map(|traj| traj.iter().map(|s| s.reward).sum::<f64>())
            .sum();
        let episodes = trajectories.len() as i64; // counts even if is_done is false.. is it okay?
        println!(
            "episodes: {:<5} avg reward per episode: {:.2}",
            episodes,
            sum_r / episodes as f64
        );

        // Train the model via policy gradient on the rollout data.
        // Flattening the result, for standard REINFORCE on non-episodic tasks.
        // Batch size is total_steps_in_all_trajectories, because we convert trajectories to flat 'rollout'.
        let batch_size: i64 = trajectories.iter().map(|traj| traj.len() as i64).sum();
        let rewards: Vec<Vec<f64>> = accumulate_rewards(&trajectories);
        let traj_flat: Trajectory = trajectories.clone().into_iter().flatten().collect();
        let reward_flat: Vec<f64> = rewards.into_iter().flatten().collect();
        let actions_flat: Vec<Action> = traj_flat.iter().map(|step| step.action.clone()).collect();
        let states_flat: Vec<SearchState> =
            traj_flat.iter().map(|step| step.state.clone()).collect();
        let reward_flat = Tensor::from_slice(&reward_flat).to_kind(Float);
        let actions_flat: Vec<i64> = actions_flat
            .iter()
            .map(|action| env.action_space.get_idx(action) as i64)
            .collect();
        let actions_flat: Tensor = Tensor::from_slice(&actions_flat);
        let observations_flat: Vec<Tensor> = states_flat
            .iter()
            .map(|state| env.state_embed(state, device))
            .collect();
        // mask logits only chosen for trajectory formation
        let sampled_mask = Tensor::zeros(
            [batch_size, env.action_space.size() as i64],
            (Float, device),
        )
        .scatter_value(1, &actions_flat.unsqueeze(1), 1.0);

        /// FIXME: When we use RL loss for RNN, how do we integrate over hidden states?
        let (logits, _) = model.forward(Tensor::stack(&observations_flat, 0), &None);
        // vanilla REINFORCE, misses baseline (critic, or historical avg)
        let log_probs =
            (sampled_mask * logits.log_softmax(1, Float)).sum_dim_intlist(1, false, Float);
        let loss: Tensor = -(reward_flat * log_probs).mean(Float);
        loss
    }
}
