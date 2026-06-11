use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs;

use parser::ast::NodeType;
use rl::action::ActionSpace;
use rl::env::Environment;
use rl::pool::Pool;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;
// Should we use tch? Is it compatible with python torch?
use tch::{Device, Tensor};

#[pyclass]
pub struct PyEnvironment {
    env: Environment,
    runtime: Runtime,
    device: Device,
}

#[pymethods]
impl PyEnvironment {
    #[new]
    fn new(
        filename: String,
        max_length: usize,
        batch_size: usize,
        use_cuda: bool,
    ) -> PyResult<Self> {
        let src =
            fs::read_to_string(&filename).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let (network, behavior_nodes) =
            parser::parser::parse(&src).map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;

        let behavior_decl = match network.nodes.get(behavior_nodes[0]) {
            Some(node) => match &node.node_type {
                NodeType::Behavior(b) => b,
                _ => return Err(PyValueError::new_err("No behavior found")),
            },
            None => return Err(PyValueError::new_err("No behavior found")),
        };

        let action_space: ActionSpace = behavior_decl.into();
        let device = if use_cuda {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };
        let mut runtime = Runtime::new(10000, "data".into(), Some(device));
        let backtester = BasicBacktest::new(&mut runtime.dmgr, "returns_next");
        let pool = Pool::new(backtester, device);

        let env = Environment::new(&network, action_space, pool, max_length, batch_size);

        Ok(Self {
            env,
            runtime,
            device,
        })
    }

    fn reset(&mut self) {
        self.env.reset();
    }

    fn action_space_size(&self) -> usize {
        self.env.action_space.size()
    }

    fn pool_size(&self) -> usize {
        self.env.pool.len()
    }

    fn step(&mut self, action_idx: usize) -> PyResult<(f64, bool)> {
        if action_idx >= self.env.action_space.size() {
            return Err(PyValueError::new_err("Invalid action index"));
        }
        let action = self.env.action_space.get_action(action_idx);
        self.env.step(&action);
        let is_done = action == rl::action::Action::Done;
        let reward = self
            .env
            .pool
            .calc_reward(&mut self.runtime, &self.env.state.machine, is_done);

        if is_done {
            let callgraph = self.env.state.machine.callgraph.clone();
            self.env.pool.insert(&mut self.runtime, callgraph);
        }

        Ok((reward, is_done))
    }

    fn get_valid_actions(&self, py: Python<'_>) -> PyResult<Py<PyArray1<bool>>> {
        let (stack, _callgraph) = self.env.state.machine.get_stack();
        let mut valid_actions = vec![false; self.env.action_space.size()];
        for action_idx in 0..self.env.action_space.size() {
            let action = self.env.action_space.get_action(action_idx);
            let valid = match &action {
                rl::action::Action::Done => stack.len() == 1,
                rl::action::Action::Reduce(op_spec) => {
                    stack.len() >= op_spec.inputs.len()
                        && self.env.state.machine.check_reduce(&op_spec)
                }
                _ => true,
            };
            valid_actions[action_idx] = valid;
        }
        Ok(ndarray::Array1::from_vec(valid_actions)
            .into_pyarray_bound(py)
            .into())
    }

    fn get_observation(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<(Py<PyArray3<f32>>, Vec<String>, String)> {
        let (stack, callgraph) = self.env.state.machine.get_stack();
        let mut data_tensors = Vec::new();

        for (_signal_decl, addr) in stack.iter() {
            let signal = self.runtime.lookup_or_run(callgraph, *addr);
            if let stdlib::types::Signal::DataFrame(Some(df)) = signal {
                // TODO: do embedding or mean_dim in the later
                // let df_mean = df.mean_dim(Some([0].as_slice()), false, tch::Kind::Float);
                data_tensors.push(df.to_device(Device::Cpu));
            }
        }

        let mut node_strs = Vec::new();
        for node in &callgraph.nodes {
            let s = match &node.node_type {
                parser::ast::NodeType::Operator(op) => op.name.to_string(),
                parser::ast::NodeType::Literal(lit) => format!("{}", lit),
                parser::ast::NodeType::Behavior(b) => {
                    b.name.clone().unwrap_or_else(|| "_".to_string())
                }
            };
            node_strs.push(s);
        }
        let expr_str = callgraph.format_node(callgraph.root);

        if data_tensors.is_empty() {
            let empty_data = ndarray::Array3::<f32>::zeros((0, 0, 0));
            return Ok((
                empty_data.into_pyarray_bound(py).into(),
                node_strs,
                expr_str,
            ));
        }

        let stacked = Tensor::stack(&data_tensors, 0).to_device(Device::Cpu);
        let size = stacked.size(); // [num_tensors, rows, cols]

        let flattened: Vec<f32> = stacked
            .flatten(0, -1)
            .try_into()
            .expect("Failed to convert stacked tensor to Vec<f32>");
        let array3 = ndarray::Array3::from_shape_vec(
            (size[0] as usize, size[1] as usize, size[2] as usize),
            flattened,
        )
        .map_err(|e: ndarray::ShapeError| PyValueError::new_err(e.to_string()))?;

        Ok((array3.into_pyarray_bound(py).into(), node_strs, expr_str))
    }
}

#[pymodule]
fn comet_env(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEnvironment>()?;
    Ok(())
}
