use crate::{action::Action, state::SearchState};
use runtime::ast::PolishExpr;

pub type Trajectory = Vec<TrajectoryItem>;
#[derive(Clone)]
pub struct TrajectoryItem {
    pub state: SearchState,
    pub action: Action,
    pub reward: f64,
    pub next_state: Option<SearchState>,
    pub sequence: PolishExpr, //For debugging
}
