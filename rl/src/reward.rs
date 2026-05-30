// let position = self.runtime.run(&self.network, *tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
// let pnl_result = self.pnl_calc.pnl(&position);
// let stats: Stats = (&pnl_result).into();
// let fitness = self.score_fn.fitness(&stats);
// loss::policy_gradient::calc_terminal_reward(fitness)

// loss::policy_gradient::calc_intermediate_reward()
