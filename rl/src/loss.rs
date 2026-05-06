/// TODO: https://github.com/benbaarber/rl/blob/main/Cargo.toml
/// https://github.com/LaurentMazare/tch-rs/tree/main/examples/reinforcement-learning

pub(crate) mod policy_gradient {

    use crate::trajectory::Trajectory;

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
}
