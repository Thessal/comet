use crate::action::Action;
use crate::env::Environment;
use crate::loss;
use crate::model::Model;
use crate::trajectory::Trajectory;
use tch::{
    Device,
    Kind::Float,
    Tensor,
    nn::{self, LSTMState, OptimizerConfig},
};

pub struct BatchConfig {
    pub batch_size: usize,
    pub trajectories: Vec<Trajectory>,
}

impl<'a> Environment<'a> {
    pub fn sample_trajectory(&mut self, model: &Model, device: Device) -> Trajectory {
        self.reset(); // resets state
        let mut lstmstate: Option<LSTMState> = None; // resets lstm hidden input 
        let mut trajectory: Trajectory = Vec::new();
        let action_space = self.action_space.clone();

        for _ in 0..self.max_length {
            let _state = self.state_embed(&self.state, device);
            let observation = _state;

            let (logits_not_masked, lstmstate_next) =
                tch::no_grad(|| model.forward(observation.unsqueeze(0), &lstmstate));
            lstmstate = lstmstate_next;
            let valid_actions: Vec<Action> = self.state.get_valid_actions(&action_space);
            let available_actions: Tensor = action_space.calculate_mask(&valid_actions);
            let is_invalid = available_actions.logical_not();
            let logits = logits_not_masked.masked_fill(&is_invalid, f64::NEG_INFINITY); // Petersen (2021)

            let sampled_actions: Vec<i64> =
                tch::no_grad(|| logits.softmax(1, Float).multinomial(1, true))
                    .try_into()
                    .unwrap();
            let action_idx = sampled_actions[0];
            let action: Action = action_space.get_action(action_idx as usize);
            let step = self.step(action); // self.state is updated here.
            let is_done = step.is_done();
            trajectory.push(step);
            if is_done {
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

    /// Trains an agent using the policy gradient algorithm.
    pub fn run(&mut self, model: &Model, device: Device) {
        let vs = nn::VarStore::new(device);
        let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();
        println!("action space: {:?}", self.action_space);
        for epoch_idx in 0..50 {
            self.sample_trajectories(model, device);
            println!("epoch: {:<3} ", epoch_idx);
            let loss = loss::policy_gradient::calculate_loss(self, model, device);
            opt.backward_step(&loss);
        }
    }
}
