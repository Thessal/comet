use std::collections::HashMap;

use parser::ast::Network;
use parser::behavior::BehaviorDecl;
use rl::env::Environment;
use rl::model::RandomModel;
use rl::pool::Pool;
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
        action_space,
        pool,
        30,  // max_length
        100, // batch_size
    );

    env.sample(
        &mut runtime,
        &mut RandomModel::new(env.action_space.clone()),
        &device,
    );

    // We create a minimal Environment
    // RewardCalculator requires Runtime and Pool, we'll assume it has a default or skip if we can't build it easily
    // But BruteforceSearch is just what the user asked. Let's just create a dummy if needed or leave brute_force empty.
}
