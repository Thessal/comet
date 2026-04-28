use std::collections::HashMap;

use crate::action::Action;
use crate::action::ActionSpace;
use crate::state::SearchState;
use parser::program::BehaviorDecl;
use runtime::ast::PolishExpr;
use runtime::ast::Tree;
use runtime::runtime::Runtime;

pub type Trajectory = Vec<(SearchState, Action, f64, bool)>; // state, action, reward, done

pub struct CometRlEnv<'a> {
    behavior: BehaviorDecl, // used for reset
    pub trajectory: Trajectory,
    pub runtime: &'a Runtime,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
}

impl<'a> CometRlEnv<'a> {
    pub fn new(runtime: &'a Runtime, behavior: BehaviorDecl, max_length: usize) -> Self {
        CometRlEnv {
            behavior: behavior.clone(),
            trajectory: vec![],
            runtime: runtime,
            state: behavior.into(),
            action_space: behavior.into(),
            max_length: max_length,
        }
    }
}

impl<'a> CometRlEnv<'a> {
    fn reset(&mut self) -> SearchState {
        self.trajectory = vec![];
        self.state = self.behavior.clone().into();
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (SearchState, f64, bool) {
        let (next_state, done) = self.state.apply_action(action);
        let expr_polish: &PolishExpr = &next_state.expr;
        let expr: Tree = expr_polish.into();
        let position = self.runtime.run(&expr);
        let pnl = self.runtime.pnl(&position);
        let fitness = self.runtime.fitness(&pnl);
        (next_state, reward, done)
    }
}
