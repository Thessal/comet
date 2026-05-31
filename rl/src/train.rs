use crate::action::Action;
use crate::env::Environment;
use crate::loss;
use crate::model::Model;
use crate::trajectory::Trajectory;
use tch::{
    Device,
    Kind::Float,
    Tensor,
    nn::{self, LSTMState, OptimizerConfig},
};

pub struct BatchConfig {
    pub batch_size: usize,
    pub max_length: usize,
    pub trajectories: Vec<Trajectory>,
}
