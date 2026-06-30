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
    adj_coeff: f64,
} // TODO: evicting pool, do not add invalid equations to the pool

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

    pub fn new(backtester: BasicBacktest, device: tch::Device, adj_coeff: f64) -> Self {
        Pool {
            asts: HashMap::new(),
            returns: HashMap::new(),
            portfolio_returns: tch::Tensor::zeros(SIGNAL_LENGTH, (tch::Kind::Float, device)),
            backtester,
            device,
            adj_coeff,
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
                f64::try_from(self.marginal_utility(&r.unsqueeze(0), 252, 0).max())
                    .unwrap_or(f64::NAN); // TODO: pass evaluation start/end days
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
    fn marginal_utility(
        &self,
        returns_stacked: &Tensor,
        evaluation_start_days: i64,
        evaluation_end_days: i64,
    ) -> Tensor {
        // returns_stacked : [*, T]
        let t_dim = returns_stacked.size()[1];
        let eval_len = t_dim - evaluation_start_days - evaluation_end_days;

        let returns_eval = returns_stacked.narrow(1, evaluation_start_days, eval_len);
        let port_eval = self
            .portfolio_returns
            .narrow(0, evaluation_start_days, eval_len);

        let pool_size = self.len() as f64;

        let fitness_old = if pool_size == 0.0 {
            tch::Tensor::zeros(&[returns_eval.size()[0]], (tch::Kind::Float, self.device))
        } else {
            let old_mean = port_eval.mean(None);
            let old_std = port_eval.std(false).clamp_min(1e-9);
            (&old_mean / &old_std).broadcast_to(&[returns_eval.size()[0]])
        };

        let returns_new = if pool_size == 0.0 {
            returns_eval.shallow_clone()
        } else {
            let weight_old = pool_size / (pool_size + 1.0);
            let weight_new = 1.0 / (pool_size + 1.0); // TODO: use optimizer, instead of uniform weight
            (&port_eval * weight_old) + (returns_eval * weight_new)
        };

        let new_mean = returns_new.mean_dim(Some([1_i64].as_slice()), false, tch::Kind::Float);
        let new_std = returns_new.std_dim(Some([1_i64].as_slice()), false, false);
        let fitness_new = &new_mean / new_std.clamp_min(1e-9);

        let marginal_utility = fitness_new - fitness_old;
        marginal_utility
    }
}

impl Pool {
    pub fn calc_reward(
        &self,
        runtime: &mut Runtime,
        machine: &AbstractMachine,
        // _is_done: bool,
        pbest_potential: f64,
    ) -> Result<(f64, f64), &'static str> {
        let (stack, callgraph): (&Vec<(Signal, usize)>, &Network) = machine.get_stack();
        if stack.is_empty() {
            return Err("Empty Stack");
        }
        let mut returns_list = Vec::with_capacity(stack.len());
        for (signal_spec, addr) in stack {
            // (types, address)
            if let Signal::DataFrame(_) = signal_spec {
                returns_list.push(self.signal_to_returns(runtime, callgraph, *addr));
            } else {
                // println!("Diagnostic - Ignoring non-dataframe signal in calc_reward: {:?}", signal_spec);
            }
        }

        if returns_list.is_empty() {
            // If the stack contains only literals, the agent hasn't built a portfolio yet.
            // Return 0.0 reward to avoid NaN computations.
            // println!("Diagnostic - returns_list is empty, returning 0.0 reward.");
            return Ok((pbest_potential, 0.0));
        }

        let returns_stacked = Tensor::stack(&returns_list, 0); // [stack_size, T]

        // Calculate marginal utility and get the maximum
        let marginal_utility = self.marginal_utility(&returns_stacked, 252, 0); // TODO: pass evaluation start/end days properly

        if bool::try_from(marginal_utility.isnan().any()).unwrap() == true {
            return Err("NaN in marginal utility.");
        }
        let potential = f64::try_from(marginal_utility.max()).unwrap();
        if potential.is_nan() || potential.is_infinite() {
            for (_signal_spec, addr) in stack {
                let incoming_pos = runtime
                    .lookup_or_run(callgraph, *addr)
                    .to_dataframe(self.device);
                let incoming_ret = self.backtester.calc_returns(&incoming_pos);
                let pos_nan_count = incoming_pos.isnan().sum(tch::Kind::Int64).double_value(&[]);
                let ret_nan_count = incoming_ret.isnan().sum(tch::Kind::Int64).double_value(&[]);
                println!(
                    "Debug Pool -> Position NaN count: {}, Returns NaN count: {}",
                    pos_nan_count, ret_nan_count
                );
                println!(
                    "Debug Pool -> Raw returns series: {:?}",
                    Vec::<f32>::try_from(incoming_ret.flatten(0, -1)).unwrap_or_default()
                );
                // let human_readable_traj: Vec<String> = actions
                //     .iter()
                //     .map(|&idx| format!("{:?}", self.action_space.get_action(idx as usize)))
                //     .collect();
                // println!("WARNING: invalid reward detected!");
                // println!(
                //     "State Machine Stack Size: {}",
                //     self.state.machine.get_stack().0.len()
                // );
                // println!("Trajectory: {:?}", human_readable_traj);
            }
            return Err("NaN in marginal utility.");
        }
        if potential.is_infinite() {
            return Err("Inf in marginal utility.");
        }
        let reward = potential - pbest_potential;
        Ok((potential, reward))
    }
}
