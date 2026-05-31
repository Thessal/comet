use crate::{action::Action, state::SearchState};

pub type Trajectory = Vec<Step>;
#[derive(Clone)]
pub struct Step {
    pub state: SearchState,
    pub action: Action,
    pub reward: f64,
    pub next_state: Option<SearchState>,
    // pub sequence: PolishExpr, //For debugging
}
impl Step {
    pub fn is_done(&self) -> bool {
        self.action == Action::Done
    }
}
