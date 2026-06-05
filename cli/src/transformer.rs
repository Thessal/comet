use parser::ast::Network;
use tch::nn::OptimizerConfig;
use rl::action::Action;
use rl::env::Environment;
use rl::model::AgentModel;
use rl::pool::Pool;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;
use tch::Device;

pub struct TransformerSearch {
    env: Environment,
}

impl TransformerSearch {
    pub fn new(env: Environment) -> Self {
        Self { env }
    }

    pub fn search(&mut self, runtime: &mut Runtime, device: &Device) {
        let vs = tch::nn::VarStore::new(*device);
        let mut model = AgentModel::new(&vs.root(), self.env.action_space.clone(), 64);
        self.env.sample(runtime, &mut model, device);
    }
}

pub fn transformer_search(network: Network, action_space: rl::action::ActionSpace, use_cuda: bool) {
    let device = if use_cuda {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };
    let mut runtime = Runtime::new(10000, "data".into(), Some(device));
    let backtester = BasicBacktest::new(&mut runtime.dmgr, "returns_next");
    let pool = Pool::new(backtester, device);

    let mut env = Environment::new(
        &network,
        action_space,
        pool,
        50,   // max_length
        1000, // batch_size
    );

    println!("Sampling trajectories...");
    let vs = tch::nn::VarStore::new(device);
    let mut model = AgentModel::new(&vs.root(), env.action_space.clone(), 64);
    let trajectories = env.sample(&mut runtime, &mut model, &device);

    // rather than beginning empty alpha pool, insert all for testing
    println!("Adding trajectories to pool...");
    trajectories.iter().for_each(|(traj, _expr, machine)| {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                env.pool.insert(&mut runtime, machine.callgraph.clone());
            }
        }
    });
    let pool_stats = env.pool.stats();

    // Calc utility for each trajectory
    println!("Calculating utility for each trajectory...");
    trajectories.iter().for_each(|(traj, expr, machine)| {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                let marginal_utility = env.pool.calc_reward(&mut runtime, &machine, true);
                let utility_traj: Vec<f64> = traj.iter().map(|step| step.reward).collect();
                let (_utility, _marginal_utility, corr, maxcorr) = pool_stats.get(expr).unwrap();
                println!(
                    "marginal_utility: {}\t utility_traj: {:?}\t Expr: {}\t _utility: {}\t _marginal_utility: {}\t corr:{}\t maxcorr:{}",
                    marginal_utility, utility_traj, expr, _utility, _marginal_utility, corr, maxcorr
                );
            } else {
                println!("Expression failed to terminate: {}", expr);
            }
        }
    });

    // todo : loss and train
    println!("Training model using PPO...");
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-4).unwrap();
    let mut total_loss_val = 0.0;
    
    // Example PPO loop (1 epoch over the collected trajectories)
    // Note: A full implementation requires batching and recalculating log_probs via model.forward()
    for (traj, _expr, machine) in trajectories.iter() {
        if let Some(last_step) = traj.last() {
            if last_step.action == Action::Done {
                let episode_return = env.pool.calc_reward(&mut runtime, &machine, true);
                
                // For each step in this trajectory
                for step in traj.iter() {
                    let advantage = episode_return - step.value;
                    let advantage_t = tch::Tensor::from_slice(&[advantage as f32]).to(device);
                    let return_t = tch::Tensor::from_slice(&[episode_return as f32]).to(device);
                    let old_value_t = tch::Tensor::from_slice(&[step.value as f32]).to(device);
                    
                    // In actual PPO, re-run forward pass here for the step state to get new log_probs and value
                    // let (new_state_emb, new_logits, new_value) = model.forward(&step.state, ...);
                    // let new_log_probs = calculate_log_prob(new_logits, step.action);
                    
                    // Dummy log_probs for compilation structure (replace with actual network outputs)
                    let dummy_log_prob = tch::Tensor::zeros([1], (tch::Kind::Float, device)).requires_grad_(true);
                    let dummy_old_log_prob = tch::Tensor::zeros([1], (tch::Kind::Float, device));
                    
                    let ppo_loss = model.calculate_ppo_loss(
                        &dummy_log_prob,
                        &dummy_old_log_prob,
                        &advantage_t,
                        None,
                        0.01,
                        0.2
                    );
                    
                    let value_loss = model.calculate_value_loss(&old_value_t, &return_t);
                    let total_loss = ppo_loss + value_loss * 0.5;
                    
                    opt.backward_step(&total_loss);
                    total_loss_val += total_loss.double_value(&[]);
                }
            }
        }
    }
    println!("PPO training step complete. Loss sum: {}", total_loss_val);
}
