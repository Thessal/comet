use crate::search::{Action, ActionSpace, SearchState};
use parser::program::BehaviorDecl;
use runtime::runtime::Runtime;

pub struct CometRlEnv {
    pub runtime: Runtime,
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub max_length: usize,
}

impl CometRlEnv {
    pub fn new(runtime: Runtime, behavior: BehaviorDecl, max_length: usize) -> Self {
        let state = SearchState { stack: vec![] };
        CometRlEnv {
            runtime: runtime, // comes with internal cache
            state: state,
            action_space: behavior.into(),
            max_length: max_length,
        }
    }
}

impl From<BehaviorDecl> for SearchState {
    fn from(x: BehaviorDecl) -> SearchState {
        SearchState {
            unprocessed_params: x.params,
            stack: vec![],
        }
    }
}

impl CometRlEnv {
    fn reset(&mut self) -> SearchState {
        self.state = SearchState {
            unprocessed_params: vec![],
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
