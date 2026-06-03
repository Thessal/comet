use std::collections::HashMap;

use parser::ast::Network;
use parser::behavior::BehaviorDecl;
use rl::action::Action;
use rl::env::Environment;
use rl::model::RandomModel;
use rl::pool::Pool;
use rl::trajectory::Step;
use runtime::backtest::BasicBacktest;
use runtime::dmgr;
use runtime::runtime::Runtime;
use tch::Device;

pub struct BruteforceSearch {
    env: Environment,
}

impl BruteforceSearch {
    pub fn new(env: Environment) -> Self {
        Self { env }
    }

    pub fn search(&mut self, runtime: &mut Runtime, device: &Device) {
        let mut model = RandomModel::new(self.env.action_space.clone());
        self.env.sample(runtime, &mut model, device);
    }
}

pub fn brute_force(
    network: Network,
    action_space: rl::action::ActionSpace,
    // behavior_decls: Vec<BehaviorDecl>,
    // behavior_nodes: Vec<usize>,
) {
    let device = Device::Cpu;
    let mut runtime = Runtime::new(10000, "data".into(), None);
    let backtester = BasicBacktest::new(&mut runtime.dmgr, "returns");
    let pool = Pool::new(backtester, device);

    let mut env = Environment::new(
        &network,
        action_space,
        pool,
        30,   // max_length
        1000, // batch_size
    );

    let trajectories = env.sample(
        &mut runtime,
        &mut RandomModel::new(env.action_space.clone()),
        &device,
    );

    // rather than beginning empty alpha pool, insert all for testing
    trajectories.iter().for_each(|(traj, expr, machine)| {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                env.pool.insert(&mut runtime, machine.callgraph.clone());
            }
        }
    });
    let pool_stats = env.pool.stats();

    // Calc utility for each trajectory
    trajectories.iter().for_each(|(traj, expr, machine)| {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                let marginal_utility = env.pool.calc_reward(&mut runtime, &machine, true);
                let utility_traj: f64 = traj.iter().map(|step| step.reward).sum::<f64>();
                let (_utility, _marginal_utility, corr) = pool_stats.get(expr).unwrap();
                println!(
                    "marginal_utility: {}\t utility_traj: {}\t Expr: {}\t _utility: {}, _marginal_utility: {}, corr:{}",
                    marginal_utility, utility_traj, expr, _utility, _marginal_utility, corr
                );
            } else {
                println!("Expression failed to terminate: {}", expr);
            }
        }
    });
}
