use std::collections::HashMap;

use parser::ast::Network;
use stdlib::types::Signal;
use tch::Tensor;

use crate::state::AbstractMachine;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;

static SIGNAL_LENGTH: i64 = stdlib::types::SIZE[0] as i64;

pub struct Pool {
    asts: HashMap<String, Network>,
    returns: HashMap<String, Tensor>,
    portfolio_returns: Tensor,
    backtester: BasicBacktest,
    device: tch::Device,
}

impl Pool {
    pub fn save_returns(&self, path: &str) {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path).expect("Unable to create file");
        for (expr, tensor) in &self.returns {
            let vals = Vec::<f64>::try_from(tensor.contiguous().to_kind(tch::Kind::Double))
                .expect("Failed to convert tensor to Vec<f64>");
            let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            let expr_escaped = expr.replace("\"", "\"\"");
            let line = format!("\"{}\",{}\n", expr_escaped, vals_str.join(","));
            file.write_all(line.as_bytes())
                .expect("Unable to write data");
        }
    }

    pub fn new(backtester: BasicBacktest, device: tch::Device) -> Self {
        Pool {
            asts: HashMap::new(),
            returns: HashMap::new(),
            portfolio_returns: tch::Tensor::zeros(SIGNAL_LENGTH, (tch::Kind::Float, device)),
            backtester,
            device,
        }
    }

    pub fn exprs(&self) -> Vec<String> {
        // self.asts.keys().cloned().collect()
        self.returns.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.returns.len()
    }

    pub fn stats(&self) -> HashMap<String, (f64, f64, f64, f64)> {
        // This function is slow. Debug / Analysis use only.
        let mut stats: HashMap<String, (f64, f64, f64, f64)> = HashMap::new();
        let max_corrs = self.maxcorr();
        for (expr, r) in self.returns.iter() {
            let utility: f64 = self.utility(r);
            let marginal_utility =
                f64::try_from(self.marginal_utility(&r.unsqueeze(0)).max()).unwrap_or(f64::NAN);
            let corr = self.corr(&self.portfolio_returns, r);
            let maxcorr = max_corrs.get(expr).copied().unwrap_or(f64::NAN);
            stats.insert(expr.clone(), (utility, marginal_utility, corr, maxcorr));
        }
        stats
    }

    fn calc_portfolio_returns(&mut self) {
        let returns: Vec<&Tensor> = self.returns.values().map(|ret| ret).collect();
        if returns.is_empty() {
            self.portfolio_returns =
                tch::Tensor::zeros(SIGNAL_LENGTH, (tch::Kind::Float, self.device));
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
            self.calc_portfolio_returns();
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

    pub fn maxcorr(&self) -> HashMap<String, f64> {
        let keys: Vec<&String> = self.returns.keys().collect();
        if keys.is_empty() {
            return HashMap::new();
        }
        if keys.len() == 1 {
            let mut map = HashMap::new();
            map.insert(keys[0].clone(), f64::NAN);
            return map;
        }

        let returns: Vec<&Tensor> = keys.iter().map(|k| self.returns.get(*k).unwrap()).collect();
        let pool_returns = tch::Tensor::stack(&returns, 0); // [N, T]
        let t = pool_returns.size()[1] as f64;

        let pool_mean = pool_returns.mean_dim(Some([1_i64].as_slice()), true, tch::Kind::Float);
        let pool_centered = &pool_returns - pool_mean;
        let pool_std = pool_centered.std_dim(Some([1_i64].as_slice()), false, true);

        let pool_norm = pool_centered / pool_std.clamp_min(1e-9);
        let corr = pool_norm.matmul(&pool_norm.transpose(0, 1)) / t; // [N, N]

        let n = keys.len() as i64;
        let eye = tch::Tensor::eye(n, (tch::Kind::Float, self.device));
        let corr = corr.masked_fill(&eye.to_kind(tch::Kind::Bool), f64::NEG_INFINITY);

        let (max_corrs, _) = corr.max_dim(1, false); // Returns (Tensor, Tensor)

        let max_corrs_vec = Vec::<f64>::try_from(max_corrs).unwrap();

        let mut result = HashMap::new();
        for (i, key) in keys.iter().enumerate() {
            result.insert((*key).clone(), max_corrs_vec[i]);
        }
        result
    }

    // Bailey, David H., and Marcos Lopez de Prado. "The Sharpe ratio efficient frontier." Journal of Risk 15.2 (2012): 13.
    // Bailey, David H., Marcos López de Prado, and Eva del Pozo. "The strategy approval decision: A sharpe ratio indifference curve approach." Algorithmic Finance 2.1 (2013): 99-109.
    fn marginal_utility(&self, returns_stacked: &Tensor) -> Tensor {
        // returns_stacked : [*, T]

        // Calculate incoming utility for all in stack
        let mean = returns_stacked.mean_dim(Some([1_i64].as_slice()), false, tch::Kind::Float);
        let std = returns_stacked.std_dim(Some([1_i64].as_slice()), false, false);
        let incoming_utility = &mean / std.clamp_min(1e-9);

        // Calculate portfolio properties
        let port_mean = self.portfolio_returns.mean(None);
        let port_std = self.portfolio_returns.std(false).clamp_min(1e-9);
        let port_utility = &port_mean / &port_std;
        let port_centered = &self.portfolio_returns - &port_mean;

        // Calculate correlation for all in stack
        let returns_centered = returns_stacked
            - returns_stacked.mean_dim(Some([1_i64].as_slice()), true, tch::Kind::Float);
        let cov = (&returns_centered * port_centered).mean_dim(
            Some([1_i64].as_slice()),
            false,
            tch::Kind::Float,
        );
        let corr = cov / (&std * &port_std).clamp_min(1e-9);

        let marginal_utility = incoming_utility - port_utility * corr;
        marginal_utility
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
            if stack.len() != 1 {
                return f64::NAN;
            }
            let (_signal_spec, addr) = &stack[0];
            let returns = self.signal_to_returns(runtime, callgraph, *addr);
            self.utility(&returns)
        } else {
            // Intermediate reward
            if stack.is_empty() {
                return f64::NAN;
            }
            let mut returns_list = Vec::with_capacity(stack.len());
            for (_signal_spec, addr) in stack {
                // (types, address)
                returns_list.push(self.signal_to_returns(runtime, callgraph, *addr));
            }
            let returns_stacked = Tensor::stack(&returns_list, 0); // [stack_size, T]

            // Calculate marginal utility and get the maximum
            let marginal_utility = self.marginal_utility(&returns_stacked);

            // Mask out NaNs to effectively ignore them in max
            let nan_mask = marginal_utility.isnan();
            let masked_marginal = marginal_utility.masked_fill(&nan_mask, f64::NEG_INFINITY);

            let max_val = f64::try_from(masked_marginal.max()).unwrap_or(f64::NAN);
            if max_val == f64::NEG_INFINITY {
                f64::NAN
            } else {
                max_val
            }
        }
    }
}
