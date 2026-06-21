use parser::ast::Network;
use rl::action::Action;
use rl::env::Environment;
use rl::model::RandomModel;
use rl::pool::Pool;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;
use tch::Device;

// pub struct BruteforceSearch {
//     env: Environment,
// }

// impl BruteforceSearch {
//     pub fn new(env: Environment) -> Self {
//         Self { env }
//     }

//     pub fn search(&mut self, runtime: &mut Runtime, device: &Device) {
//         let mut model = RandomModel::new(self.env.action_space.clone(), 0.1, 0.5);
//         self.env.sample(runtime, &mut model, device);
//     }
// }

pub fn brute_force(
    network: Network,
    action_space: rl::action::ActionSpace,
    use_cuda: bool,
) -> Pool {
    let device = if use_cuda {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };
    let mut runtime = Runtime::new(10000, "data".into(), Some(device));
    let backtester = BasicBacktest::new(&mut runtime.dmgr, "returns_next");
    let pool = Pool::new(backtester, device, 1.0);

    let mut env = Environment::new(
        &network,
        action_space,
        pool,
        50, // max_length
        50, // batch_size (episodes_per_batch)
    );

    let mut model = RandomModel::new(
        env.action_space.clone(),
        0.02, // introduce_prob
        1.0,  // stop_prob
    );
    let num_iterations = 2000;

    for iteration in 0..num_iterations {
        println!("--- Iteration {} ---", iteration);
        let trajectories = env.sample(&mut runtime, &mut model, &device);

        let mut ep_lengths = Vec::new();

        trajectories.iter().for_each(|(traj, _expr, machine)| {
            ep_lengths.push(traj.len());
            if let Some(last_step) = traj.last() {
                if last_step.action == Action::Done {
                    env.pool.insert(&mut runtime, machine.callgraph.clone());
                }
            }
        });

        let avg_length: f64 = ep_lengths.iter().sum::<usize>() as f64 / ep_lengths.len() as f64;
        println!(
            "Avg Length: {:.1} | Pool Size: {}",
            avg_length,
            env.pool.len()
        );
    }

    env.pool
}
