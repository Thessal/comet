use std::collections::HashMap;

use crate::action::Action;
use crate::action::ActionSpace;
use crate::state::SearchState;
use crate::train::BatchConfig;
use parser::behavior::BehaviorDecl;
use runtime::ast::PolishExpr;
use runtime::ast::Tree;
use runtime::pnl;
use runtime::runtime::Runtime;
use runtime::stats::{Aggregator, Stats};

pub type Trajectory = Vec<(SearchState, Action, f64, bool)>; // state, action, reward, done

pub struct Environment<'a> {
    pub behavior: BehaviorDecl, // used for reset
    pub trajectory: Trajectory,
    pub config: BatchConfig,
    pub runtime: &'a mut Runtime,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
    pub pnl_calc: pnl::PnlCalculator<'a>,
    pub score_fn: Aggregator,
}

impl<'a> Environment<'a> {
    pub fn new(
        runtime: &'a mut Runtime,
        behavior: BehaviorDecl,
        score_fn: Aggregator,
        max_length: usize,
    ) -> Self {
        Environment {
            behavior: behavior.clone(),
            trajectory: vec![],
            runtime: runtime,
            state: behavior.clone().into(),
            action_space: behavior.clone().into(),
            max_length: max_length,
            pnl_calc: pnl::PnlCalculator::new(&mut runtime.dmgr),
            score_fn: score_fn,
        }
    }
}

impl<'a> Environment<'a> {
    fn reset(&mut self) -> SearchState {
        self.trajectory = vec![];
        self.state = self.behavior.clone().into();
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (SearchState, f64, bool) {
        let (next_state, done) = self.state.clone().apply_action(action);
        self.state = next_state.clone();
        let expr_polish: &PolishExpr = &next_state.expr;
        let expr: Tree = expr_polish.into();
        let position = self.runtime.run(&expr);
        let pnl_result = self.pnl_calc.pnl(&position);
        let stats: Stats = (&pnl_result).into();
        let fitness = self.score_fn.fitness(&stats);
        (next_state, fitness, done)
    }
}
