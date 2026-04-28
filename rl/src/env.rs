use crate::search::{Action, ActionSpace, SearchState};
use parser::program::BehaviorDecl;
use runtime::runtime::Runtime;

pub struct CometRlEnv {
    behavior: BehaviorDecl,
    pub runtime: &Runtime,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
}

impl CometRlEnv {
    pub fn new(runtime: &Runtime, behavior: BehaviorDecl, max_length: usize) -> Self {
        CometRlEnv {
            behavior: behavior.clone(),
            runtime: runtime,
            state: behavior.into(),
            action_space: behavior.into(),
            max_length: max_length,
        }
    }
}

impl From<BehaviorDecl> for SearchState {
    fn from(x: BehaviorDecl) -> SearchState {
        SearchState {
            params: todo!(), //x.args.iter().map(|arg| arg.name.clone()).collect(),
            stack: vec![],
        }
    }
}

impl CometRlEnv {
    fn reset(&mut self) -> SearchState {
        self.state = SearchState {
            params: vec![],
            stack: vec![],
        };
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (SearchState, f64, bool) {
        let (next_state, done) = self.state.apply_action(action);
        let sim_result = self.runtime.evaluate_sequence(&next_state.sequence);
        let reward = sim_result.fitness;
        (next_state, reward, done)
    }
}
