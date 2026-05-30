use crate::action::Action;
use crate::action::ActionSpace;
use crate::state::SearchState;
use crate::train::BatchConfig;
use parser::ast::Network;
use parser::behavior::BehaviorDecl;

pub struct Environment {
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub config: BatchConfig,
    orig_call_graph_size: usize, //network_size
    orig_behavior_addr: usize,   //node_idx
    orig_behavior: BehaviorDecl, //behavior_decl
}

impl Environment {
    pub fn new(
        call_graph: &Network,
        action_space: ActionSpace,
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
