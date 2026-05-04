use std::collections::HashMap;

use crate::action::Action;
use crate::action::ActionSpace;
use crate::model::ModelConfig;
use crate::state::SearchState;
use crate::train::BatchConfig;
use crate::trajectory::Trajectory;
use crate::trajectory::TrajectoryItem;
use burn::tensor::backend::Backend;
use parser::behavior::BehaviorDecl;
use runtime::ast::PolishExpr;
use runtime::ast::Tree;
use runtime::pnl;
use runtime::runtime::Runtime;
use runtime::stats::{Aggregator, Stats};

pub struct Environment<'a> {
    pub behavior: BehaviorDecl, // used for reset
    pub trajectory: Trajectory,
    pub config: BatchConfig,
    pub runtime: &'a mut Runtime,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
    pub pnl_calc: pnl::PnlCalculator,
    pub score_fn: Aggregator,
}

impl<'a> Environment<'a> {
    pub fn new(
        runtime: &'a mut Runtime,
        behavior: BehaviorDecl,
        score_fn: Aggregator,
        max_length: usize,
        batch_size: usize,
    ) -> Self {
        let pnl_calc = pnl::PnlCalculator::new(&mut runtime.dmgr);
        Environment {
            behavior: behavior.clone(),
            trajectory: vec![],
            config: BatchConfig {
                batch_size,
                trajectories: vec![],
            },
            runtime: runtime,
            state: behavior.clone().into(),
            action_space: (&behavior).into(),
            max_length: max_length,
            pnl_calc,
            score_fn: score_fn,
        }
    }
}

impl<'a> Environment<'a> {
    pub fn reset(&mut self) -> SearchState {
        self.trajectory = vec![];
        self.state = self.behavior.clone().into();
        self.state.clone()
    }

    pub fn step(&mut self, action: Action) -> TrajectoryItem {
        let (next_state, done) = self.state.apply_action(action.clone(), &mut self.runtime);
        let reward = match done {
            false => {
                assert!(
                    next_state
                        .stack
                        .iter()
                        .all(|(_expr, _tree, data)| !data.is_none())
                );
                todo!("calc_intermediate_reward()"); // batch diversity
            }
            true => {
                // let expr: Tree = expr_polish.into();
                assert!(next_state.stack.len() == 1);
                let (_expr, tree, _data) = next_state.stack.get(0).unwrap();
                let position = self.runtime.run(tree);
                let pnl_result = self.pnl_calc.pnl(&position);
                let stats: Stats = (&pnl_result).into();
                let fitness = self.score_fn.fitness(&stats);
                todo!("calc_terminal_reward(fitness)");
            }
        };

        let traj_item = TrajectoryItem {
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
    use parser::expr::Ident;
    use parser::{behavior::NamedSignal, expr::Literal};
    use runtime::ast::{OperatorSpec, Program, Token};
    use runtime::runtime::test_make_param0;
    use runtime::stats::Metric;
    use stdlib::types::Signal;

    use crate::model::RnnModelConfig;

    use super::*;

    #[test]
    fn test_environment_step() {
        let mut runtime = Runtime::new(100, "../data".into());
        // let param0 = test_make_param0();
        let behavior = test_make_behavior();
        let score_fn = Aggregator {
            weights: HashMap::from_iter([
                (Metric::Sharpe, (0.5, 0., 1.)),
                (Metric::Ret, (0.5, 0., 1.)),
            ]),
        };
        let action_space: ActionSpace = (&behavior).into();
        let model = ModelConfig::RnnModel(RnnModelConfig {
            action_vocab_size: action_space.size(),
            d_model: 10,
            d_hidden: 10,
            dropout: 0.1,
        });
        let mut env = Environment::new(&mut runtime, behavior, score_fn, 10, 100);

        let actions = vec![
            Action::ShiftParam(0), //data
            Action::ShiftInt(5),   //5
            Action::Reduce(OperatorSpec::from("ts_mean")),
        ];
        let mut done = false;

        for a in actions {
            assert_eq!(done, false);
            let TrajectoryItem {
                state,
                action,
                reward,
                next_state,
                sequence,
            } = env.step(a);
            done = matches!(action, Action::Done);
            println!("Reward : {:?}", reward);
        }
    }
}
