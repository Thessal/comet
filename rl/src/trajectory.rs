use crate::{action::Action, state::SearchState};
use tch::Tensor;

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
