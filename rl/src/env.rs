use crate::action::Action;
use crate::action::ActionSpace;
use crate::loss;
use crate::model::LstmModel;
use crate::model::Model;
use crate::reward;
use crate::reward::RewardCalculator;
use crate::state::SearchState;
use crate::train::BatchConfig;
use crate::trajectory::Step;
use crate::trajectory::Trajectory;
use parser::ast::Network;
use parser::behavior::BehaviorDecl;
use runtime::runtime::Runtime;
use stdlib::types::Signal;
use tch::IndexOp;
use tch::{
    Device,
    Kind::Float,
    Tensor,
    nn::{self, LSTMState, OptimizerConfig},
};

pub struct Environment {
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub config: BatchConfig,
    reward_calculator: RewardCalculator,
    orig_call_graph_size: usize, //network_size
    orig_behavior_addr: usize,   //node_idx
    orig_behavior: BehaviorDecl, //behavior_decl
}

impl Environment {
    pub fn new(
        call_graph: &Network,
        action_space: ActionSpace,
        reward_calculator: RewardCalculator,
        max_length: usize,
        batch_size: usize,
    ) -> Self {
        let (behavior_idx, behavior_ref) = call_graph.get_behavior();

        let result = Self {
            state: SearchState::new(call_graph),
            action_space: action_space,
            config: BatchConfig {
                max_length,
                batch_size,
                trajectories: vec![],
            },
            reward_calculator,
            orig_call_graph_size: call_graph.nodes.len(),
            orig_behavior_addr: behavior_idx,
            orig_behavior: behavior_ref.clone(),
        };
        result
    }
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn step(&mut self, action: &Action) {
        self.state.apply_action(action);
    }
}

impl Environment {
    pub fn sample<T: Model>(&mut self, runtime: &Runtime, model: &mut T, device: &Device) {
        for _ in 0..self.config.batch_size {
            self.sample_one(runtime, model, device);
        }
    }

    fn get_valid_action_mask(&self, device: &Device) -> Tensor {
        let (stack, callgraph) = self.state.machine.get_stack();
        let mut valid_actions: Vec<Action> = vec![];
        for action_idx in 0..self.action_space.size() {
            let action: Action = self.action_space.get_action(action_idx);
            let valid = match &action {
                Action::Done => stack.len() == 1,
                Action::Reduce(op_spec) => {
                    stack.len() >= op_spec.inputs.len() && // stack size checking
                    self.state.machine.check_reduce(&op_spec) // type checking
                }
                _ => {
                    // introducing new variable is always valid
                    true
                }
            };
            if valid {
                valid_actions.push(action);
            }
        }
        self.action_space.calculate_mask(&valid_actions)
    }

    fn sample_one<T: Model>(
        &mut self,
        runtime: &Runtime,
        model: &mut T,
        device: &Device,
    ) -> Trajectory {
        self.reset();
        let mut trajectory: Trajectory = Vec::new();
        for i in 0..self.config.max_length {
            let mask: Tensor = self.get_valid_action_mask(device);
            let (state_embedding, action_logits): (Tensor, Tensor) =
                model.forward(&self.state, &mask, device);
            let sampled_action_idx: Vec<Vec<i64>> = // [batch_size=1, sample_number=1]
                tch::no_grad(|| action_logits.softmax(1, Float).multinomial(1, true))
                    .try_into()
                    .unwrap();
            let action_idx: i64 = sampled_action_idx[0][0];
            let action: Action = self.action_space.get_action(action_idx as usize);
            self.step(&action);

            let reward: f64 = self.reward_calculator.calc_reward(&self.state.machine);

            let done = action == Action::Done;
            let step = Step {
                state_embedding,
                action,
                reward,
                next_state_embedding: None,
            };

            trajectory.push(step);
            if done {
                break;
            }
        }
        trajectory
    }

    // pub fn sample(&mut self, model: &Model, device: Device) -> Trajectory {
    //     self.reset(); // resets state
    //     for _ in 0..self.config.max_length {
    //         let mut lstmstate: Option<LSTMState> = None; // resets lstm hidden input
    //         let mut trajectory: Trajectory = Vec::new();
    //         let action_space = self.action_space.clone();

    //         let _state = self.state_embed(&self.state, device);
    //         let observation = _state;

    //         let (logits_not_masked, lstmstate_next) =
    //             tch::no_grad(|| model.forward(observation.unsqueeze(0), &lstmstate));
    //         lstmstate = lstmstate_next;

    //         let valid_actions: Vec<Action> = self.state.get_valid_actions(&action_space);
    //         if valid_actions.is_empty() {
    //             break;
    //         }
    //         let available_actions: Tensor = action_space.calculate_mask(&valid_actions);
    //         let is_invalid = available_actions.logical_not();
    //         let logits = logits_not_masked.masked_fill(&is_invalid, f64::NEG_INFINITY); // Petersen (2021)
    //         assert_eq!(
    //             logits.size(),
    //             [1, action_space.size() as i64] // [batch_size, action_vocab_size]
    //         );

    //         let sampled_actions: Vec<Vec<i64>> = // [batch_size, single sample]
    //             tch::no_grad(|| logits.softmax(1, Float).multinomial(1, true))
    //                 .try_into()
    //                 .unwrap();
    //         let action_idx = sampled_actions[0][0];
    //         let action: Action = action_space.get_action(action_idx as usize);
    //         let step = self.step(action); // self.state is updated here.
    //         let is_done = step.is_done();
    //         trajectory.push(step);
    //         if is_done {
    //             break;
    //         }
    //     }
    //     trajectory
    // }

    // pub fn sample_trajectories(&mut self, model: &Model, device: Device) {
    //     self.config.trajectories.clear();
    //     for _ in 0..self.config.batch_size {
    //         let traj = self.sample_trajectory(model, device);
    //         self.config.trajectories.push(traj);
    //     }
    // }

    // /// Trains an agent using the policy gradient algorithm.
    // pub fn run(&mut self, model: &Model, epochs: usize, device: Device) {
    //     let vs = nn::VarStore::new(device);
    //     let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();
    //     println!("action space: {:?}", self.action_space);
    //     for epoch_idx in 0..epochs {
    //         self.sample_trajectories(model, device);
    //         println!("epoch: {:<3} ", epoch_idx);
    //         println!(
    //             "trajectory count in epoch : {}",
    //             self.config.trajectories.len()
    //         );
    //         println!(
    //             "finished equations : {}",
    //             self.config
    //                 .trajectories
    //                 .iter()
    //                 .filter(|t| t.last().unwrap().is_done())
    //                 .count()
    //         );
    //         println!(
    //             "example : {:?}",
    //             self.config.trajectories[0]
    //                 .iter()
    //                 .last()
    //                 .unwrap()
    //                 .state
    //                 .expr
    //         );
    //         println!(
    //             "eqn length - max: {}, min:{}",
    //             self.config
    //                 .trajectories
    //                 .iter()
    //                 .map(|x| x.len())
    //                 .max()
    //                 .unwrap(),
    //             self.config
    //                 .trajectories
    //                 .iter()
    //                 .map(|x| x.len())
    //                 .min()
    //                 .unwrap()
    //         );
    //         println!("cache size : {}", self.runtime.expr_cache.len());
    //         let loss = loss::policy_gradient::calculate_loss(self, model, device);
    //         opt.backward_step(&loss);
    //     }
    // }
}
