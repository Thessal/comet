use crate::search::{Action, SearchEnv, SearchState};
use runtime::runtime::Runtime;

/// Burn-RL style Environment Interface
pub trait RlEnvironment {
    fn reset(&mut self) -> SearchState;
    fn step(&mut self, action: Action) -> (SearchState, f64, bool); // state, reward, done
}

pub struct CometRlEnv {
    pub runtime: Runtime,
    pub state: SearchState,
    pub search_env: SearchEnv,
    pub available_funcs: Vec<(
        String,
        Vec<parser::ast::TypeDecl>,
        parser::ast::TypeDecl,
    )>,
    pub max_length: usize,
    pub param_names: Vec<String>,
}

impl CometRlEnv {
    pub fn new(
        target: parser::ast::TypeDecl,
        funcs: Vec<(
            String,
            Vec<parser::ast::TypeDecl>,
            parser::ast::TypeDecl,
        )>,
        param_names: Vec<String>,
        cache_capacity: usize,
    ) -> Self {
        CometRlEnv {
            runtime: Runtime::new(cache_capacity, "data"),
            state: SearchState {
                unprocessed_params: vec![],
                stack: vec![],
                sequence: vec![],
            },
            search_env: SearchEnv::new(target),
            available_funcs: funcs,
            max_length: 50,
            param_names,
        }
    }
}

impl RlEnvironment for CometRlEnv {
    fn reset(&mut self) -> SearchState {
        self.state = SearchState {
            unprocessed_params: vec![],
            stack: vec![],
            sequence: vec![],
        };
        self.state.clone()
    }

    fn step(&mut self, action: Action) -> (SearchState, f64, bool) {
        if self.state.sequence.len() >= self.max_length {
            return (self.state.clone(), -1.0, true); // Penalize hitting max depth
        }

        match self
            .search_env
            .step(&self.state, action, &self.available_funcs)
        {
            Ok(new_state) => {
                self.state = new_state;

                // Done?
                if self.state.unprocessed_params.is_empty()
                    && self.state.stack.len() == 1
                    && self.state.stack[0] == self.search_env.target_return
                {
                    // Natively evaluate Stack sequence without intermediate AST/DAG compilation
                    match self.runtime.evaluate_sequence(
                        &self.state.sequence,
                        self.param_names.clone(),
                    ) {
                        Ok(_output) => {
                            // Sequence execution successful! 
                            // Normally calculate fitness or cross entropy here
                            (self.state.clone(), 1.0, true)
                        }
                        Err(_) => {
                            // Map translation failed
                            (self.state.clone(), -1.0, true)
                        }
                    }
                } else {
                    // Intermediate step
                    (self.state.clone(), 0.0, false)
                }
            }
            Err(_) => {
                // Invalid action taken
                (self.state.clone(), -1.0, true)
            }
        }
    }
}

pub fn get_available_funcs() -> Vec<(String, Vec<parser::ast::TypeDecl>, parser::ast::TypeDecl)> {
    let mut funcs = Vec::new();
    for meta in inventory::iter::<stdlib::OperatorMeta> {
        let name = meta.name.to_string();
        if name == "data" {
            continue;
        }
        let mut inputs = Vec::new();
        for input in meta.inputs {
            inputs.push(match input {
                stdlib::OutputShape::DataFrame => parser::ast::TypeDecl::DataFrame,
                stdlib::OutputShape::TimeSeries => parser::ast::TypeDecl::DataFrame,
                stdlib::OutputShape::Vector => parser::ast::TypeDecl::Vector,
                stdlib::OutputShape::Matrix => parser::ast::TypeDecl::Matrix,
                stdlib::OutputShape::Void => parser::ast::TypeDecl::Void,
                stdlib::OutputShape::ScalarFloat => parser::ast::TypeDecl::Float,
                stdlib::OutputShape::ScalarInt => parser::ast::TypeDecl::Float,
                stdlib::OutputShape::ScalarString => parser::ast::TypeDecl::String,
            });
        }
        let output = match meta.output_shape {
            stdlib::OutputShape::DataFrame => parser::ast::TypeDecl::DataFrame,
            stdlib::OutputShape::TimeSeries => parser::ast::TypeDecl::DataFrame,
            stdlib::OutputShape::Vector => parser::ast::TypeDecl::Vector,
            stdlib::OutputShape::Matrix => parser::ast::TypeDecl::Matrix,
            stdlib::OutputShape::Void => parser::ast::TypeDecl::Void,
            stdlib::OutputShape::ScalarFloat => parser::ast::TypeDecl::Float,
            stdlib::OutputShape::ScalarInt => parser::ast::TypeDecl::Float,
            stdlib::OutputShape::ScalarString => parser::ast::TypeDecl::String,
        };
        funcs.push((name, inputs, output));
    }
    // Sort consistently
    funcs.sort_by(|a, b| a.0.cmp(&b.0));
    funcs
}

