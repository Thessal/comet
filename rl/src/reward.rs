// let position = self.runtime.run(&self.network, *tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
// let pnl_result = self.pnl_calc.pnl(&position);
// let stats: Stats = (&pnl_result).into();
// let fitness = self.score_fn.fitness(&stats);
// loss::policy_gradient::calc_terminal_reward(fitness)

// loss::policy_gradient::calc_intermediate_reward()

use std::collections::HashMap;

use parser::ast::Network;
use stdlib::types::Signal;
use tch::Tensor;

use crate::state::AbstractMachine;
use crate::state::SearchState;
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
    runtime: Runtime,
    asts: HashMap<String, Network>,
    returns: HashMap<String, Tensor>,
    portfolio_returns: Tensor,
    backtester: Backtester,
}

impl Pool {
    fn new(runtime: Runtime, backtester: Backtester) -> Self {
        Pool {
            runtime,
            asts: HashMap::new(),
            returns: HashMap::new(),
            portfolio_returns: tch::Tensor::zeros(
                SIGNAL_LENGTH,
                (tch::Kind::Float, tch::Device::Cpu),
            ),
            backtester,
        }
    }

    fn calc_portfolio_returns(&mut self) {
        let returns: Vec<Tensor> = self.returns.values().map(|ret| ret.clone()).collect();
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

    fn insert(&mut self, sub_ast: Network) {
        let hash_str: String = sub_ast.format_node(0);
        let pos = self.runtime.lookup_or_run(&sub_ast, 0);
        let returns = self.backtester.calc_returns(pos);
        self.asts.insert(hash_str.clone(), sub_ast);
        self.returns.insert(hash_str, returns);
    }

    fn utility(&self, returns: &Tensor) -> f64 {
        let sharpe = returns.mean(None) / returns.std(false);
        f64::try_from(sharpe).unwrap_or(0.0)
    }

    // Bailey, David H., and Marcos Lopez de Prado. "The Sharpe ratio efficient frontier." Journal of Risk 15.2 (2012): 13.
    // Bailey, David H., Marcos López de Prado, and Eva del Pozo. "The strategy approval decision: A sharpe ratio indifference curve approach." Algorithmic Finance 2.1 (2013): 99-109.
    fn marginal_utility(&mut self, callgraph: &Network, root: usize) -> f64 {
        let incoming_pos = self.runtime.lookup_or_run(callgraph, root);
        let incoming_ret = self.backtester.calc_returns(incoming_pos);

        // Calculate correlation manually if corrcoef is not easily accessible
        let cov = (&self.portfolio_returns - self.portfolio_returns.mean(None))
            * (&incoming_ret - incoming_ret.mean(None));
        let corr_tensor =
            cov.mean(None) / (self.portfolio_returns.std(false) * incoming_ret.std(false));
        let corr = f64::try_from(corr_tensor).unwrap_or(0.0);

        let incoming_utility = self.utility(&incoming_ret) * corr;
        let portfolio_utility = self.utility(&self.portfolio_returns);
        let marginal_utility = incoming_utility - portfolio_utility;
        marginal_utility
    }
}

pub struct RewardCalculator {
    runtime: Runtime,
    pool: Pool,
}

impl RewardCalculator {
    pub fn calc_reward(&mut self, machine: &AbstractMachine) -> f64 {
        let (stack, callgraph): (&Vec<(Signal, usize)>, &Network) = machine.get_stack();
        // stack (types, address)

        let mut marginal_utility_all = vec![];
        for (_signal_spec, addr) in stack {
            let utility = self.pool.marginal_utility(callgraph, *addr);
            marginal_utility_all.push(utility);
        }

        if marginal_utility_all.is_empty() {
            return 0.0;
        }

        *marginal_utility_all
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}
