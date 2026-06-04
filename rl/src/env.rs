use crate::action::Action;
use crate::action::ActionSpace;
use crate::model::Model;
use crate::pool::Pool;
use crate::state::AbstractMachine;
use crate::state::SearchState;
use parser::ast::Network;
use runtime::runtime::Runtime;
use tch::{Device, Kind::Float, Tensor};

pub type Trajectory = Vec<Step>;
pub struct Step {
    pub state_embedding: Tensor,
    pub action: Action,
    pub reward: f64,
    pub next_state_embedding: Option<Tensor>,
    // pub sequence: PolishExpr, //For debugging
}
impl Step {
    pub fn is_done(&self) -> bool {
        self.action == Action::Done
    }
}

pub struct BatchConfig {
    pub batch_size: usize,
    pub max_length: usize,
    pub trajectories: Vec<Trajectory>,
}

pub struct Environment {
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub config: BatchConfig,
    pub pool: Pool,
}

impl Environment {
    pub fn new(
        call_graph: &Network,
        action_space: ActionSpace,
        pool: Pool,
        max_length: usize,
        batch_size: usize,
    ) -> Self {
        let result = Self {
            state: SearchState::new(call_graph),
            action_space: action_space,
            config: BatchConfig {
                max_length,
                batch_size,
                trajectories: vec![],
            },
            pool,
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
    pub fn sample<T: Model>(
        &mut self,
        runtime: &mut Runtime,
        model: &mut T,
        device: &Device,
    ) -> Vec<(Trajectory, String, AbstractMachine)> {
        let mut trajectories = vec![];
        for _ in 0..self.config.batch_size {
            let (steps, expr) = self.sample_one(runtime, model, device);
            let machine = self.state.machine.clone();
            trajectories.push((steps, expr, machine));
        }
        trajectories
    }

    fn get_valid_action_mask(&self, device: &Device) -> Tensor {
        let (stack, _callgraph) = self.state.machine.get_stack();
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
        self.action_space.calculate_mask(&valid_actions).to(*device)
    }

    fn sample_one<T: Model>(
        &mut self,
        runtime: &mut Runtime,
        model: &mut T,
        device: &Device,
    ) -> (Trajectory, String) {
        self.reset();
        let mut trajectory: Trajectory = Vec::new();
        for _i in 0..self.config.max_length {
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

            let is_done = action == Action::Done;
            let reward: f64 = self.pool.calc_reward(runtime, &self.state.machine, is_done);

            let step = Step {
                state_embedding,
                action,
                reward,
                next_state_embedding: None,
            };

            trajectory.push(step);
            if is_done {
                break;
            }
        }
        let expr = self
            .state
            .machine
            .callgraph
            .format_node(self.state.machine.callgraph.root);
        (trajectory, expr)
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
