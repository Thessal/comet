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
    root: usize,
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
        root,
        action_space,
        pool,
        30, // max_length
        10, // batch_size
    );

    let trajectories = env.sample(
        &mut runtime,
        &mut RandomModel::new(env.action_space.clone()),
        &device,
    );

    // rather than beginning empty alpha pool, insert all for testing
    trajectories.iter().for_each(|(traj, expr, machine)| {
        if let Some(last_step) = traj.last() {
            env.pool.insert(&mut runtime, machine.callgraph.clone());
        }
    });

    // calc utility for each trajectory
    trajectories.iter().for_each(|(traj, expr, machine)| {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                let utility = env.pool.calc_reward(&mut runtime, &machine, true);
                println!("Utility: {}\t Expr: {}", utility, expr);
            } else {
                println!("Expression failed to terminate: {}", expr);
            }
        }
    });
}
