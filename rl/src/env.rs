use crate::action::Action;
use crate::action::ActionSpace;
use crate::embed::data_embedding_model;
use crate::loss;
use crate::state::SearchState;
use crate::train::BatchConfig;
use crate::trajectory::Step;
use parser::ast::{Network, Node, NodeType};
use parser::behavior::BehaviorDecl;
use runtime::pnl;
use runtime::runtime::Runtime;
use runtime::stats::{Aggregator, Stats}; // todo : store returns matrix inside stats struct
use tch::Tensor;

pub struct Environment<'a> {
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub call_graph: Network,
    pub config: BatchConfig,
    pub orig_behavior: (usize, usize, BehaviorDecl), // node_idx, network_size, behavior_decl
}

impl<'a> Environment<'a> {
    pub fn new(
        call_graph: &Network,
        action_space: ActionSpace,
        max_length: usize,
        batch_size: usize,
    ) -> Self {
        let (behavior_idx, behavior_ref) = call_graph.get_behavior();

        let mut result = Self {
            state: SearchState::new_dummy(),
            action_space: action_space,
            call_graph: call_graph.clone(),
            config: BatchConfig {
                max_length,
                batch_size,
                trajectories: vec![],
            },
            orig_behavior: (behavior_idx, call_graph.nodes.len(), behavior_ref.clone()),
        };
        result.reset();
        result
    }

    pub fn reset(&mut self) {
        let (behavior_idx, network_size, orig_behavior) = &self.orig_behavior;
        // reset search state
        self.state.stack = vec![];
        // reset behavior node
        self.state.callgraph.nodes[*behavior_idx].node_type =
            NodeType::Behavior(orig_behavior.clone());
        self.state.callgraph.nodes.truncate(*network_size);
    }

    pub fn step(&mut self, action: &Action) -> Step {
        self.state.apply_action(action);
        let reward = match action {
            Action::Done => {
                // let expr: Tree = expr_polish.into();
                assert!(next_state.stack.len() == 1);
                let (_expr, tree, _data) = next_state.stack.get(0).unwrap();
                let position = self.runtime.run(&self.network, *tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
                let pnl_result = self.pnl_calc.pnl(&position);
                let stats: Stats = (&pnl_result).into();
                let fitness = self.score_fn.fitness(&stats);
                loss::policy_gradient::calc_terminal_reward(fitness)
            }
            _ => {
                assert!(
                    next_state
                        .stack
                        .iter()
                        .all(|(_expr, _tree, data)| !data.is_none())
                );
                loss::policy_gradient::calc_intermediate_reward()
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

// #[cfg(test)]
// mod tests {
//     use parser::behavior::test_make_behavior;
//     use runtime::ast::Program;
//     use runtime::runtime::test_make_param0;
//     use runtime::stats::Metric;
//     use std::collections::HashMap;
//     use stdlib::OperatorSpec;

//     // use crate::model::ModelConfig;
//     // use crate::model::RnnModelConfig;

//     use super::*;

//     #[test]
//     fn test_environment_step() {
//         let mut runtime = Runtime::new(100, "../data".into(), None);
//         let param0: Program = test_make_param0();
//         let behavior = test_make_behavior();
//         let score_fn = Aggregator {
//             weights: HashMap::from_iter([
//                 (Metric::Sharpe, (0.5, 0., 1.)),
//                 (Metric::Ret, (0.5, 0., 1.)),
//             ]),
//         };
//         // let action_space: ActionSpace = (&behavior).into();
//         // let model = ModelConfig::RnnModel(RnnModelConfig {
//         //     action_vocab_size: action_space.size(),
//         //     d_model: 10,
//         //     d_hidden: 10,
//         //     dropout: 0.1,
//         // });
//         let mut env = Environment::new(
//             &mut runtime,
//             behavior,
//             vec![param0.into()],
//             score_fn,
//             10,
//             100,
//         );

//         let actions = vec![
//             Action::ShiftParam(0), //data
//             Action::ShiftInt(5),   //5
//             Action::Reduce(OperatorSpec::from("ts_mean")),
//         ];
//         let mut done = false;

//         for a in actions {
//             assert_eq!(done, false);
//             let Step {
//                 state: _state,
//                 action,
//                 reward,
//                 next_state: _next_state,
//                 sequence: _sequence,
//             } = env.step(a);
//             done = matches!(action, Action::Done);
//             println!("Reward : {:?}", reward);
//         }
//     }

//     #[test]
//     fn test_action_finish() {
//         let mut runtime = Runtime::new(100, "../data".into(), None);
//         let param0: Program = test_make_param0();
//         let behavior = test_make_behavior();
//         let score_fn = Aggregator {
//             weights: HashMap::from_iter([
//                 (Metric::Sharpe, (0.5, 0., 1.)),
//                 (Metric::Ret, (0.5, 0., 1.)),
//             ]),
//         };

//         let mut env = Environment::new(
//             &mut runtime,
//             behavior.clone(),
//             vec![param0.into()],
//             score_fn,
//             10,
//             1, // batch size
//         );

//         // Setup almost finished environment
//         let actions = vec![
//             Action::ShiftParam(0),
//             Action::ShiftInt(5),
//             Action::Reduce(OperatorSpec::from("ts_mean")),
//         ];

//         for a in actions {
//             env.step(a);
//         }

//         // 1. Inspect valid actions
//         let valid_actions = env.state.get_valid_actions(&env.action_space);
//         println!(
//             "Valid actions at almost finished state: {:?}",
//             valid_actions
//         );
//         assert!(
//             valid_actions.contains(&Action::Done),
//             "Done should be valid"
//         );

//         // 2. Check Action Space masking
//         let mask = env.action_space.calculate_mask(&valid_actions);
//         let done_idx = env.action_space.get_idx(&Action::Done) as i64;

//         let is_done_valid = bool::try_from(mask.get(done_idx)).unwrap_or(false);
//         assert!(is_done_valid, "Mask for Done should be true");

//         // 3. (Optional) Inspect Model logits / probabilities
//         // let vs = tch::nn::VarStore::new(tch::Device::Cpu);
//         // let model_config = crate::model::ModelSize::Small.get_config(env.action_space.size());
//         // let model = crate::model::Model::RnnModel(model_config.init(&vs.root()));
//         // ... (forward pass and inspect mask)
//     }

//     //TODO: tests for invalid actions, terminal states, and maximum trajectory length boundaries.
// }
