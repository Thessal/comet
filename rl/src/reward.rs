// let position = self.runtime.run(&self.network, *tree); // FIXME: this can be slow. maybe we have to cahnge Signal::DataFrame(Vec<Vec<f64>>) into Signal::DataFrame(tch::Tensor)
// let pnl_result = self.pnl_calc.pnl(&position);
// let stats: Stats = (&pnl_result).into();
// let fitness = self.score_fn.fitness(&stats);
// loss::policy_gradient::calc_terminal_reward(fitness)

// loss::policy_gradient::calc_intermediate_reward()

use parser::ast::Network;
use stdlib::types::Signal;
use tch::Tensor;

use crate::state::AbstractMachine;
use crate::state::SearchState;
use runtime::runtime::Runtime;

fn utility(hash_list: Vec<String>) -> f64 {
    // calculate utility of solution set
    todo!()
}

pub struct RewardCalculator {
    runtime: Runtime,
    solution_set: Vec<(String, Tensor)>,
}

impl RewardCalculator {
    pub fn calc_reward(&self, machine: &AbstractMachine) -> f64 {
        let (stack, callgraph): (&Vec<(Signal, usize)>, &Network) = machine.get_stack();
        // stack (types, address)

        let mut incoming_signal: Vec<(String, Tensor)> = Vec::new();
        for (signal_spec, addr) in stack {
            let signal: &Signal = self.runtime.lookup_or_run(callgraph, *addr);
            let df: Tensor = signal.to_dataframe(device);
            let hash: String = callgraph.format_node(*addr);
            incoming_signal.push((hash, df));
        }

        let utility_before = utility(self.solution_set);
        self.solution_set.extend(incoming_signal);
        let utility_after = utility(self.solution_set);
        self.solution_set
            .truncate(self.solution_set.len() - incoming_signal.len());

        let reward = utility_after - utility_before;
        reward
    }
}

// // cast literals into tensor
// // assert tensor size
// // For intermediate reward
// // * Measure data entropy / symbolic entropy

// // Schmidt and Lipson
// let parsimony = 1.0/((i+1) as f64);
// let stack_size = stack.len() as f64;

// // * Measure data performance (how?)
// let predictive_accuracy = todo!(); //out-sample error

// // * Let's read the Polya's book, "How to Solve it."
// // For final reward
// // * compute performance of result
// // compute reward
// /// We need to think before implement this part....

// runtime.lookup_or_run(&self.state.call_graph, &self.state.expr);
// let (_, reward) = runtime.run(&self.state);
