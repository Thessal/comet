// let position = self.runtime.run(&self.network, *tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
// let pnl_result = self.pnl_calc.pnl(&position);
// let stats: Stats = (&pnl_result).into();
// let fitness = self.score_fn.fitness(&stats);
// loss::policy_gradient::calc_terminal_reward(fitness)

// loss::policy_gradient::calc_intermediate_reward()

use std::collections::HashMap;

use parser::ast::Network;
use parser::behavior::BehaviorDecl;
use stdlib::types::Signal;
use tch::Tensor;

use crate::state::AbstractMachine;
use crate::state::SearchState;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;

static SIGNAL_LENGTH: i64 = stdlib::types::SIZE[0] as i64;
// pub struct Backtester {}

// impl Backtester {
//     pub fn new() -> Self {
//         Backtester {}
//     }

//     pub fn calc_returns(&self, signal: &Signal) -> Tensor {
//         // Dummy implementation of calc_returns
//         // In a real scenario, this would compute financial returns from a Signal
//         tch::Tensor::ones(
//             [SIGNAL_LENGTH].as_slice(),
//             (tch::Kind::Float, tch::Device::Cpu),
//         )
//     }
// }

pub struct Pool {
    asts: HashMap<String, Network>,
    returns: HashMap<String, Tensor>,
    portfolio_returns: Tensor,
    backtester: BasicBacktest,
    device: tch::Device,
}

impl Pool {
    pub fn new(backtester: BasicBacktest, device: tch::Device) -> Self {
        Pool {
            asts: HashMap::new(),
            returns: HashMap::new(),
            portfolio_returns: tch::Tensor::zeros(
                SIGNAL_LENGTH,
                (tch::Kind::Float, tch::Device::Cpu),
            ),
            backtester,
            device,
        }
    }

    pub fn exprs(&self) -> Vec<String> {
        // self.asts.keys().cloned().collect()
        self.returns.keys().cloned().collect()
    }

    pub fn stats(&self) -> HashMap<String, (f64, f64, f64)> {
        let mut stats: HashMap<String, (f64, f64, f64)> = HashMap::new();
        for (expr, r) in self.returns.iter() {
            let utility: f64 = self.utility(r);
            let marginal_utility = self.marginal_utility(r);
            let corr = self.corr(&self.portfolio_returns, r);
            stats.insert(expr.clone(), (utility, marginal_utility, corr));
        }
        stats
    }

    fn calc_portfolio_returns(&mut self) {
        let returns: Vec<&Tensor> = self.returns.values().map(|ret| ret).collect();
        if returns.is_empty() {
            self.portfolio_returns =
                tch::Tensor::zeros(SIGNAL_LENGTH, (tch::Kind::Float, tch::Device::Cpu));
        } else {
            self.portfolio_returns = tch::Tensor::stack(&returns, 0).mean_dim(
                Some([0_i64].as_slice()),
                false,
                tch::Kind::Float,
            );
        }
    }

    pub fn insert(&mut self, runtime: &mut Runtime, sub_ast: Network) {
        // you can use Network::extract_subtree to get subtrees
        let hash_str: String = sub_ast.format_node(sub_ast.root);
        if !self.asts.contains_key(&hash_str) {
            let pos = runtime.lookup_or_run(&sub_ast, sub_ast.root);
            let returns = self.backtester.calc_returns(&pos.to_dataframe(self.device));
            self.asts.insert(hash_str.clone(), sub_ast);
            self.returns.insert(hash_str, returns);
        }
    }

    fn utility(&self, returns: &Tensor) -> f64 {
        let mean = f64::try_from(returns.mean(None)).unwrap();
        let std = f64::try_from(returns.std(false)).unwrap();
        let val = mean / std.max(1e-9);
        // if val.is_nan() { 0.0 } else { val }
        val
    }

    fn signal_to_returns(&self, runtime: &mut Runtime, callgraph: &Network, root: usize) -> Tensor {
        let incoming_pos = runtime
            .lookup_or_run(callgraph, root)
            .to_dataframe(self.device);
        let incoming_ret = self.backtester.calc_returns(&incoming_pos);
        incoming_ret
    }

    fn corr(&self, r1: &Tensor, r2: &Tensor) -> f64 {
        assert_eq!(r1.size(), r2.size());
        let cov = (r1 - r1.mean(None)) * (r2 - r2.mean(None));
        let cov = f64::try_from(cov.mean(None)).unwrap();
        let std = f64::try_from(r1.std(false) * r2.std(false)).unwrap();
        let std = std.max(1e-9);
        cov / std
        // let corr = f64::try_from(corr_tensor).unwrap_or(0.0);
        // if corr.is_nan() { 0.0 } else { corr }
    }

    // Bailey, David H., and Marcos Lopez de Prado. "The Sharpe ratio efficient frontier." Journal of Risk 15.2 (2012): 13.
    // Bailey, David H., Marcos López de Prado, and Eva del Pozo. "The strategy approval decision: A sharpe ratio indifference curve approach." Algorithmic Finance 2.1 (2013): 99-109.
    fn marginal_utility(&self, incoming_ret: &Tensor) -> f64 {
        // Calculate correlation manually if corrcoef is not easily accessible
        let corr = self.corr(&self.portfolio_returns, incoming_ret);
        let incoming_utility = self.utility(&incoming_ret);
        let portfolio_utility = self.utility(&self.portfolio_returns);
        assert!(!corr.is_nan());
        assert!(!incoming_utility.is_nan());
        assert!(!portfolio_utility.is_nan());
        let marginal_utility = incoming_utility - portfolio_utility * corr;
        if marginal_utility.is_nan() {
            f64::NAN
        } else {
            marginal_utility
        }
    }
}

impl Pool {
    pub fn calc_reward(
        &self,
        runtime: &mut Runtime,
        machine: &AbstractMachine,
        is_done: bool,
    ) -> f64 {
        let (stack, callgraph): (&Vec<(Signal, usize)>, &Network) = machine.get_stack();
        if is_done {
            // Terminal reward
            // TODO: refer runtime::stats
            // let's simply use sharpe for now
            assert!(stack.len() == 1);
            let (_signal_spec, addr) = &stack[0];
            let returns = self.signal_to_returns(runtime, callgraph, *addr);
            self.utility(&returns)
        } else {
            // Intermediate reward
            // stack (types, address)

            // Calculate marginal utility for each root in stack
            let mut marginal_utility_all = vec![];
            for (_signal_spec, addr) in stack {
                let returns = self.signal_to_returns(runtime, callgraph, *addr);
                let marginal_utility = self.marginal_utility(&returns);
                marginal_utility_all.push(marginal_utility);
            }

            marginal_utility_all
                .into_iter()
                .filter(|x| !x.is_nan())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f64::NAN)
        }
    }
}
