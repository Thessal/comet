use crate::action::Action;
use crate::action::ActionSpace;
use crate::embed::data_embedding_model;
use crate::loss;
use crate::state::SearchState;
use crate::train::BatchConfig;
use crate::trajectory::Step;
use parser::behavior::BehaviorDecl;
use runtime::ast::Program;
use runtime::ast::Token;
use runtime::ast::Tree;
use runtime::pnl;
use runtime::runtime::Runtime;
use runtime::stats::{Aggregator, Stats};
use tch::Tensor;

pub struct Environment<'a> {
    pub behavior: BehaviorDecl, // used for reset
    pub config: BatchConfig,
    pub runtime: &'a mut Runtime,
    pub params: Vec<Tree>,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
    pub pnl_calc: pnl::PnlCalculator,
    pub score_fn: Aggregator,
}

static EMBEDDING_SIZE_PER_TOKEN: usize = 5; // 5 floats per token
static EMBEDDING_TOKEN_CNT: usize = 2; // two tokens
pub static EMBEDDING_SIZE: usize = EMBEDDING_SIZE_PER_TOKEN * EMBEDDING_TOKEN_CNT;

impl Environment<'_> {
    pub fn state_embed(&self, state: &SearchState, device: tch::Device) -> tch::Tensor {
        // Petersen(2021): last two token, 1 float per token.
        let mut embedding_tokens: Vec<Tensor> = vec![
            Tensor::from_slice(&[0.0f64; EMBEDDING_SIZE_PER_TOKEN]),
            Tensor::from_slice(&[0.0f64; EMBEDDING_SIZE_PER_TOKEN]),
        ];
        for (i, tok) in state.expr.iter().take(EMBEDDING_TOKEN_CNT).enumerate() {
            embedding_tokens[i] = runtime::ast::token_to_tensor(tok); // FIXME
        }
        assert!(embedding_tokens.len() == EMBEDDING_TOKEN_CNT);
        let out = tch::Tensor::cat(&embedding_tokens, 0).to_device(device);
        assert!(out.dim() == 1);
        assert!(out.size()[0] as usize == EMBEDDING_SIZE);
        out

        // // TODO:
        // // SNIP (2023) paper used tokenization and attention pooling.
        // // This is simplified, max pooling based embedding. Let's try this first.
        // let data_size = self.runtime.dmgr.data_size;
        // let embeddings: Vec<Vec<Vec<f64>>> = state
        //     .stack
        //     .iter()
        //     .map(|(_, _, signal)| signal.to_dataframe(data_size))
        //     .collect();
        // todo!("data_embedding_model need to be implemented");
        // let embeddings: Vec<tch::Tensor> = embeddings
        //     .into_iter()
        //     .map(|x| data_embedding_model(&x).to_device(device))
        //     .collect();
        // tch::Tensor::stack(&embeddings, 1).max_dim(1, false).0 // max pooling
    }
}

impl<'a> Environment<'a> {
    pub fn new(
        runtime: &'a mut Runtime,
        behavior: BehaviorDecl,
        params: Vec<Tree>,
        score_fn: Aggregator,
        max_length: usize,
        batch_size: usize,
    ) -> Self {
        let pnl_calc = pnl::PnlCalculator::new(&mut runtime.dmgr);
        let trees: Vec<Tree> = params;
        // let params: Vec<Signal> = trees.iter().map(|t| runtime.run(t)).collect();
        Environment {
            behavior: behavior.clone(),
            // trajectory: vec![],
            config: BatchConfig {
                batch_size,
                trajectories: vec![],
            },
            params: trees,
            runtime: runtime,
            state: behavior.clone().into(),
            action_space: (&behavior).into(),
            max_length: max_length,
            pnl_calc,
            score_fn: score_fn,
        }
    }
    pub fn reset(&mut self) -> SearchState {
        // self.trajectory = vec![];
        self.state = self.behavior.clone().into();
        self.state.clone()
    }

    pub fn step(&mut self, action: Action) -> Step {
        let (next_state, done) =
            self.state
                .apply_action(action.clone(), &mut self.runtime, &self.params);
        let reward = match done {
            false => {
                // println!("Intermediate Expr (not done) : {:?}", next_state.expr);
                // println!("Intermediate State (not done) : {:?}", next_state.stack);
                // println!(
                //     "stack: {:?}",
                //     next_state
                //         .stack
                //         .iter()
                //         .map(|x| x.2.clone())
                //         .collect::<Vec<_>>()
                // );
                assert!(
                    next_state
                        .stack
                        .iter()
                        .all(|(_expr, _tree, data)| !data.is_none())
                );
                loss::policy_gradient::calc_intermediate_reward()
            }
            true => {
                // let expr: Tree = expr_polish.into();
                assert!(next_state.stack.len() == 1);
                let (_expr, tree, _data) = next_state.stack.get(0).unwrap();
                let position = self.runtime.run(tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
                let pnl_result = self.pnl_calc.pnl(&position);
                let stats: Stats = (&pnl_result).into();
                let fitness = self.score_fn.fitness(&stats);
                loss::policy_gradient::calc_terminal_reward(fitness)
            }
        };

        let traj_item = Step {
            state: self.state.clone(),
            action,
            reward,
            next_state: Some(next_state.clone()),
            sequence: next_state.expr.clone(),
        };

        self.state = next_state.clone();

        traj_item
    }
}
#[cfg(test)]
mod tests {
    use parser::behavior::test_make_behavior;
    use runtime::ast::Program;
    use runtime::runtime::test_make_param0;
    use runtime::stats::Metric;
    use std::collections::HashMap;
    use stdlib::OperatorSpec;

    // use crate::model::ModelConfig;
    // use crate::model::RnnModelConfig;

    use super::*;

    #[test]
    fn test_environment_step() {
        let mut runtime = Runtime::new(100, "../data".into(), None);
        let param0: Program = test_make_param0();
        let behavior = test_make_behavior();
        let score_fn = Aggregator {
            weights: HashMap::from_iter([
                (Metric::Sharpe, (0.5, 0., 1.)),
                (Metric::Ret, (0.5, 0., 1.)),
            ]),
        };
        // let action_space: ActionSpace = (&behavior).into();
        // let model = ModelConfig::RnnModel(RnnModelConfig {
        //     action_vocab_size: action_space.size(),
        //     d_model: 10,
        //     d_hidden: 10,
        //     dropout: 0.1,
        // });
        let mut env = Environment::new(
            &mut runtime,
            behavior,
            vec![param0.into()],
            score_fn,
            10,
            100,
        );

        let actions = vec![
            Action::ShiftParam(0), //data
            Action::ShiftInt(5),   //5
            Action::Reduce(OperatorSpec::from("ts_mean")),
        ];
        let mut done = false;

        for a in actions {
            assert_eq!(done, false);
            let Step {
                state: _state,
                action,
                reward,
                next_state: _next_state,
                sequence: _sequence,
            } = env.step(a);
            done = matches!(action, Action::Done);
            println!("Reward : {:?}", reward);
        }
    }

    #[test]
    fn test_action_finish() {
        let mut runtime = Runtime::new(100, "../data".into(), None);
        let param0: Program = test_make_param0();
        let behavior = test_make_behavior();
        let score_fn = Aggregator {
            weights: HashMap::from_iter([
                (Metric::Sharpe, (0.5, 0., 1.)),
                (Metric::Ret, (0.5, 0., 1.)),
            ]),
        };

        let mut env = Environment::new(
            &mut runtime,
            behavior.clone(),
            vec![param0.into()],
            score_fn,
            10,
            1, // batch size
        );

        // Setup almost finished environment
        let actions = vec![
            Action::ShiftParam(0),
            Action::ShiftInt(5),
            Action::Reduce(OperatorSpec::from("ts_mean")),
        ];

        for a in actions {
            env.step(a);
        }

        // 1. Inspect valid actions
        let valid_actions = env.state.get_valid_actions(&env.action_space);
        println!(
            "Valid actions at almost finished state: {:?}",
            valid_actions
        );
        assert!(
            valid_actions.contains(&Action::Done),
            "Done should be valid"
        );

        // 2. Check Action Space masking
        let mask = env.action_space.calculate_mask(&valid_actions);
        let done_idx = env.action_space.get_idx(&Action::Done) as i64;

        let is_done_valid = bool::try_from(mask.get(done_idx)).unwrap_or(false);
        assert!(is_done_valid, "Mask for Done should be true");

        // 3. (Optional) Inspect Model logits / probabilities
        // let vs = tch::nn::VarStore::new(tch::Device::Cpu);
        // let model_config = crate::model::ModelSize::Small.get_config(env.action_space.size());
        // let model = crate::model::Model::RnnModel(model_config.init(&vs.root()));
        // ... (forward pass and inspect mask)
    }

    //TODO: tests for invalid actions, terminal states, and maximum trajectory length boundaries.
}
