use crate::action::{Action, ActionSpace};
use crate::backtest::BasicBacktest;
use crate::dmgr::DataManager;
use lru::LruCache;
use parser::ast::{Network, Node, NodeType};
use parser::behavior::BehaviorDecl;
use parser::expr::Literal;
use std::collections::HashMap;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use stdlib::OperatorSpec;
use stdlib::types::Signal;
use tch::{Device, Kind, Tensor};

pub struct Runtime {
    pub dmgr: DataManager,
    pub expr_cache: LruCache<String, Signal>,
    pub expr_lookups: usize,
    pub expr_hits: usize,
    pub enable: bool,
}

impl Runtime {
    pub fn new(capacity: usize, data_dir: PathBuf, device: Option<Device>) -> Self {
        Runtime {
            dmgr: DataManager::new(data_dir, device),
            expr_cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()),
            expr_lookups: 0,
            expr_hits: 0,
            enable: true,
        }
    }

    pub fn lookup_or_run(&mut self, callgraph: &Network, root: usize) -> &Signal {
        let hash_key: String = callgraph.format_node(root);
        if self.expr_cache.get(&hash_key).is_none() {
            let data = self.run(callgraph, root);
            self.expr_cache.put(hash_key.clone(), data);
        }
        self.expr_cache.get(&hash_key).unwrap()
    }

    fn data_operator(&mut self, network: &Network, node: &Node) -> Signal {
        let child_sig = self.run(network, node.children[0]);
        if let Signal::String(Some(s)) = child_sig {
            Signal::DataFrame(Some(
                self.dmgr
                    .get_data(&s)
                    .expect("Data not found")
                    .shallow_clone(),
            ))
        } else {
            panic!("data operator requires a String argument");
        }
    }

    fn run(&mut self, network: &Network, root: usize) -> Signal {
        let node = &network.nodes[root];
        match &node.node_type {
            NodeType::Operator(spec) => {
                if spec.name == "data" {
                    self.data_operator(network, node)
                } else {
                    let args: Vec<Signal> = node
                        .children
                        .iter()
                        .map(|&child| self.lookup_or_run(network, child).clone())
                        .collect();
                    self.execute(spec, args).unwrap()
                }
            }
            NodeType::Behavior(_) => panic!("Behavior node cannot be run"),
            NodeType::Literal(Literal::Boolean(_)) => panic!("Boolean literal not supported"),
            NodeType::Literal(Literal::Integer(literal)) => Signal::Int(Some(*literal)),
            NodeType::Literal(Literal::Float(literal)) => Signal::Float(Some(*literal)),
            NodeType::Literal(Literal::String(literal)) => Signal::String(Some(literal.clone())),
        }
    }

    fn execute(&self, spec: &OperatorSpec, args: Vec<Signal>) -> Result<Signal, String> {
        spec.execute(&args)
    }
}

#[derive(Debug, Clone)]
pub struct AbstractMachine {
    params: Vec<(Signal, usize)>,
    stack: Vec<(Signal, usize)>,
    pub callgraph: Network,
    callgraph_behavior_idx: usize,
}

impl AbstractMachine {
    pub fn get_stack(&self) -> (&Vec<(Signal, usize)>, &Network) {
        (&self.stack, &self.callgraph)
    }

    pub fn check_reduce(&self, operator_spec: &OperatorSpec) -> bool {
        assert!(
            operator_spec.inputs.len() <= self.stack.len(),
            "Stack length is not enough. Stack: {:?}\nAST:\n{}, Operator name: {}, Operator inputs: {:?}",
            self.stack,
            self.callgraph,
            operator_spec.name,
            operator_spec.inputs,
        );
        for (op_input, stack_item) in operator_spec.inputs.iter().zip(self.stack.iter().rev()) {
            if std::mem::discriminant(op_input) != std::mem::discriminant(&stack_item.0) {
                return false;
            }
        }
        true
    }

    fn get_param_addr(&mut self, action: &Action) -> Option<usize> {
        match action {
            Action::ShiftInt(i) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::Integer(*i))),
            ),
            Action::ShiftFloat(f) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::Float(*f))),
            ),
            Action::ShiftString(s) => Some(
                self.callgraph
                    .add_node(NodeType::Literal(Literal::String(s.clone()))),
            ),
            Action::ShiftParam(p) => {
                let (_signal_type, addr) = &self.params[*p];
                Some(*addr)
            }
            Action::Reduce(op) => {
                let idx = self.callgraph.add_node(NodeType::Operator(op));
                let arity = op.inputs.len();
                for (_, addr) in self.stack.iter().rev().take(arity) {
                    self.callgraph.add_child(idx, *addr);
                }
                Some(idx)
            }
            Action::Done => None,
        }
    }

    fn push(&mut self, action: &Action) {
        let addr_incoming = self.get_param_addr(action);
        if let Action::Reduce(op) = action {
            assert!(self.check_reduce(op));
            self.stack.truncate(self.stack.len() - op.inputs.len());
        }
        let signal_incoming = self.action_to_signal(action);
        if let Some(addr) = addr_incoming {
            self.stack.push((signal_incoming, addr));
        }
    }

    fn action_to_signal(&self, action: &Action) -> Signal {
        match action {
            Action::ShiftInt(i) => Signal::Int(Some(*i)),
            Action::ShiftFloat(f) => Signal::Float(Some(*f)),
            Action::ShiftString(s) => Signal::String(Some(s.clone())),
            Action::ShiftParam(i) => self.params[*i].0.clone(),
            Action::Reduce(op_spec) => op_spec.output_shape.clone(),
            Action::Done => unreachable!(),
        }
    }
}

impl From<Network> for AbstractMachine {
    fn from(callgraph: Network) -> Self {
        let (behavior_idx, behavior_decl) = callgraph.get_behavior();
        Self {
            params: behavior_decl
                .inputs
                .iter()
                .zip(callgraph.nodes[behavior_idx].children.iter())
                .map(|(sig, &addr)| (sig.clone(), addr))
                .collect(),
            stack: vec![],
            callgraph: callgraph.clone(),
            callgraph_behavior_idx: behavior_idx,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchState {
    pub machine: AbstractMachine,
    orig_callgraph_size: usize,
    orig_behavior_node: Node,
}

impl SearchState {
    pub fn new(callgraph: &Network) -> Self {
        let machine: AbstractMachine = callgraph.clone().into();
        let behavior_idx = machine.callgraph_behavior_idx;
        let orig_behavior_node = callgraph.nodes[behavior_idx].clone();
        Self {
            machine,
            orig_callgraph_size: callgraph.nodes.len(),
            orig_behavior_node,
        }
    }

    pub fn reset(&mut self) {
        self.machine.stack.clear();
        self.machine
            .callgraph
            .nodes
            .truncate(self.orig_callgraph_size);
        self.machine.callgraph.nodes[self.machine.callgraph_behavior_idx] =
            self.orig_behavior_node.clone();
    }

    pub fn apply_action(&mut self, action: &Action) {
        match action {
            Action::Done => {
                assert!(
                    self.machine.stack.len() == 1,
                    "Stack length is not 1. Stack: {:?}\nAST:\n{}",
                    self.machine.stack,
                    self.machine.callgraph
                );
                let (subgraph_output_type, subgraph_root_idx) = self.machine.stack.last().unwrap();
                assert!(matches!(subgraph_output_type, Signal::DataFrame(_)));
                let root_node = self.machine.callgraph.nodes[*subgraph_root_idx].clone();
                self.machine.callgraph.nodes[self.machine.callgraph_behavior_idx] = root_node;
            }
            a => {
                self.machine.push(a);
            }
        }
    }
}

pub struct StackEnv {
    pub state: SearchState,
    pub action_space: ActionSpace,
    pub runtime: Runtime,
    pub backtester: BasicBacktest,
    pub device: Device,
    pub rows: i64,
    pub cols: i64,
    pub target_field: String,
    pub metric: String,
    pub epochs: usize,
    pub weights_path: Option<String>,
    pub datadir: String,
    pub max_stack_len: usize,
}

impl StackEnv {
    pub fn from_file(
        filename: &str,
        use_cuda: bool,
        cache_capacity: usize,
        normalize_book: bool,
        max_weight: Option<f64>,
    ) -> Self {
        let device = if use_cuda {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };

        let file_path = if std::path::Path::new(filename).exists() {
            PathBuf::from(filename)
        } else {
            let parent_try = PathBuf::from("..").join(filename);
            if parent_try.exists() {
                parent_try
            } else {
                PathBuf::from(filename)
            }
        };

        let src = fs::read_to_string(&file_path)
            .unwrap_or_else(|e| panic!("Failed to read .cm file {:?}: {}", file_path, e));
        let (network, behavior_nodes) = parser::parser::parse(&src)
            .unwrap_or_else(|e| panic!("Failed to parse {:?}: {}", file_path, e));

        let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
            NodeType::Behavior(b) => b,
            _ => unreachable!(),
        };
        let action_space: ActionSpace = behavior_decl.into();

        let raw_datadir = behavior_decl
            .datadir
            .clone()
            .unwrap_or_else(|| "./data".to_string());
        let datadir = if std::path::Path::new(&raw_datadir).join("info.json").exists() {
            raw_datadir
        } else {
            let parent_data = PathBuf::from("..").join(&raw_datadir);
            if parent_data.join("info.json").exists() {
                parent_data.to_string_lossy().into_owned()
            } else if let Some(cm_root) = file_path.parent().and_then(|p| p.parent()) {
                let cm_data = cm_root.join(&raw_datadir);
                if cm_data.join("info.json").exists() {
                    cm_data.to_string_lossy().into_owned()
                } else {
                    raw_datadir
                }
            } else {
                raw_datadir
            }
        };

        let info_str =
            fs::read_to_string(format!("{}/info.json", datadir)).expect("Failed to read info.json");
        let info: serde_json::Value =
            serde_json::from_str(&info_str).expect("Failed to parse info.json");
        let target_field = info["target"]
            .as_str()
            .expect("Missing 'target' in info.json")
            .to_string();
        let rows = info["rows"].as_i64().expect("Missing 'rows' in info.json");
        let cols = info["columns"]
            .as_i64()
            .expect("Missing 'columns' in info.json");

        stdlib::types::set_size(rows, cols);

        let mut runtime = Runtime::new(cache_capacity, datadir.clone().into(), Some(device));
        let state = SearchState::new(&network);
        let epochs = behavior_decl.epochs.unwrap_or(100);
        let metric = behavior_decl
            .metric
            .clone()
            .unwrap_or_else(|| "utility".to_string());
        let weights_path = behavior_decl.weights.clone();

        let backtester = BasicBacktest::new(
            &mut runtime.dmgr,
            &target_field,
            &metric,
            normalize_book,
            max_weight,
        );

        Self {
            state,
            action_space,
            runtime,
            backtester,
            device,
            rows,
            cols,
            target_field,
            metric,
            epochs,
            weights_path,
            datadir,
            max_stack_len: 5,
        }
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn step(&mut self, action_idx: usize) -> bool {
        let action = self.action_space.get_action(action_idx);
        let is_done = action == Action::Done;
        self.state.apply_action(&action);
        is_done
    }

    pub fn valid_action_mask(&self) -> Vec<bool> {
        let (stack, _callgraph) = self.state.machine.get_stack();
        let mut mask = vec![false; self.action_space.size()];
        for action_idx in 0..self.action_space.size() {
            let action = self.action_space.get_action(action_idx);
            mask[action_idx] = match &action {
                Action::Done => stack.len() == 1 && matches!(stack[0].0, Signal::DataFrame(_)),
                Action::Reduce(op_spec) => {
                    stack.len() >= op_spec.inputs.len() && self.state.machine.check_reduce(op_spec)
                }
                Action::ShiftInt(_) | Action::ShiftFloat(_) => {
                    !stack.is_empty() && (stack.len() < self.max_stack_len)
                }
                Action::ShiftParam(_) | Action::ShiftString(_) => stack.len() < self.max_stack_len,
            };
        }
        mask
    }

    pub fn stack_dataframes(&mut self) -> Vec<Tensor> {
        let (stack, callgraph) = self.state.machine.get_stack();
        let mut dfs = Vec::new();
        for (sig_spec, addr) in stack.iter() {
            if let Signal::DataFrame(_) = sig_spec {
                let df = self
                    .runtime
                    .lookup_or_run(callgraph, *addr)
                    .to_dataframe(self.device);
                dfs.push(df);
            }
        }
        dfs
    }

    pub fn evaluate_stack(&mut self) -> Vec<Vec<f32>> {
        let dfs = self.stack_dataframes();
        dfs.into_iter()
            .map(|df| {
                let ret = self
                    .backtester
                    .calc_returns(&df)
                    .to_device(Device::Cpu)
                    .to_kind(Kind::Float);
                Vec::<f32>::try_from(ret.flatten(0, -1)).unwrap_or_default()
            })
            .collect()
    }

    pub fn evaluate_top(&mut self) -> Option<Vec<f32>> {
        let dfs = self.stack_dataframes();
        dfs.last().map(|df| {
            let ret = self
                .backtester
                .calc_returns(df)
                .to_device(Device::Cpu)
                .to_kind(Kind::Float);
            Vec::<f32>::try_from(ret.flatten(0, -1)).unwrap_or_default()
        })
    }

    pub fn alpha_matrix(&mut self) -> Vec<f32> {
        let dfs = self.stack_dataframes();
        if dfs.is_empty() {
            return vec![0.0f32; (self.rows * self.cols) as usize];
        }
        let stacked = Tensor::stack(&dfs, 0);
        let mean_df = stacked
            .mean_dim(Some([0].as_slice()), false, Kind::Float)
            .nan_to_num(0.0, 0.0, 0.0)
            .to_device(Device::Cpu);
        Vec::<f32>::try_from(mean_df.flatten(0, -1)).unwrap_or_default()
    }

    pub fn expr(&self) -> String {
        let (stack, callgraph) = self.state.machine.get_stack();
        if let Some((_, addr)) = stack.last() {
            callgraph.format_node(*addr)
        } else {
            callgraph.format_node(callgraph.root)
        }
    }

    pub fn action_name(&self, idx: usize) -> String {
        let act = self.action_space.get_action(idx);
        (&act).into()
    }

    pub fn class_matrix(&self) -> Vec<f32> {
        let t = self.action_space.get_class_matrix(&Device::Cpu);
        Vec::<f32>::try_from(t.flatten(0, -1)).unwrap_or_default()
    }

}
