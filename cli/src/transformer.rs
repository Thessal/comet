use parser::ast::Network;
use rand::seq::SliceRandom;
use rl::action::Action;
use rl::env::Environment;
use rl::model::{AgentModel, Model};
use rl::pool::Pool;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;
use tch::Device;
use tch::Tensor;
use tch::nn::OptimizerConfig;

pub struct RolloutBuffer {
    pub states: Vec<(Tensor, Tensor, i64, Tensor)>, // shifted_tgt, data_pt, step_idx, mask
    pub actions: Vec<i64>,
    pub log_probs: Vec<f32>,
    pub values: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
}

impl RolloutBuffer {
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            log_probs: Vec::new(),
            values: Vec::new(),
            rewards: Vec::new(),
            dones: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.values.clear();
        self.rewards.clear();
        self.dones.clear();
    }

    pub fn calc_gae(&self, last_value: f32, gamma: f32, tau: f32) -> (Vec<f32>, Vec<f32>) {
        let mut returns = vec![0.0; self.rewards.len()];
        let mut advantages = vec![0.0; self.rewards.len()];
        let mut last_gae_lam = 0.0;
        let mut last_v = last_value;

        for t in (0..self.rewards.len()).rev() {
            let next_non_terminal = if self.dones[t] { 0.0 } else { 1.0 };
            let delta = self.rewards[t] + gamma * last_v * next_non_terminal - self.values[t];
            last_gae_lam = delta + gamma * tau * next_non_terminal * last_gae_lam;
            advantages[t] = last_gae_lam;
            returns[t] = advantages[t] + self.values[t];
            last_v = self.values[t];
        }

        (returns, advantages)
    }
}

pub struct TransformerSearch {
    env: Environment,
}

impl TransformerSearch {
    pub fn new(env: Environment) -> Self {
        Self { env }
    }

    pub fn search(&mut self, _runtime: &mut Runtime, _device: &Device) {
        // Obsolete wrapper method
    }
}

pub fn transformer_search(
    network: Network,
    action_space: rl::action::ActionSpace,
    use_cuda: bool,
) -> rl::pool::Pool {
    let device = if use_cuda {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };
    let mut runtime = Runtime::new(10000, "data".into(), Some(device));
    let backtester = BasicBacktest::new(&mut runtime.dmgr, "returns_next");
    let pool = Pool::new(backtester, device);

    let seq_len = 50;
    let mut env = Environment::new(
        &network,
        action_space.clone(),
        pool,
        seq_len, // max_length
        1,       // batch_size
    );

    let vs = tch::nn::VarStore::new(device);
    let mut model = AgentModel::new(&vs.root(), env.action_space.clone(), 256);
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-4).unwrap();
    let mut buffer = RolloutBuffer::new();

    let clip_param = 0.2;
    //  loss = policy_loss + value_loss * c1 - entropy * c2;
    let c1 = 0.5;
    let c2 = 0.05;
    let epochs = 4;
    let batch_size = 512;
    let episodes_per_batch = 50;
    let num_iterations = 2000;
    // let num_iterations = 20;
    let unfinished_penaly: f64 = 10.0;

    for iteration in 0..num_iterations {
        println!("--- Iteration {} ---", iteration);
        buffer.clear();

        let mut ep_rewards = Vec::new();
        let mut ep_lengths = Vec::new();
        let mut ep_exprs = Vec::new();

        let mut last_value_for_gae = 0.0;

        for _ep in 0..episodes_per_batch {
            let mut actions = Vec::new();
            let mut ep_reward = 0.0;
            let mut is_done = false;

            env.reset();
            let mut step_count = 0;

            for _step in 0..seq_len {
                let mask: Tensor = env.get_valid_action_mask(&device);

                // Get predictions from model
                let (alpha_matrix, action_logits, value_tensor) =
                    model.forward(&env.state, &mut runtime, &mask, &device, &actions);

                // Sample action
                let log_probs = action_logits.log_softmax(-1, tch::Kind::Float);
                let mut probs = log_probs.exp().nan_to_num(0.0, 0.0, 0.0).clamp(0.0, 1.0);
                let sum = probs.sum(tch::Kind::Float);
                if sum.double_value(&[]) <= 1e-8 {
                    probs = mask.unsqueeze(0).to_kind(tch::Kind::Float);
                    let mask_sum = probs.sum(tch::Kind::Float);
                    probs = &probs / mask_sum;
                } else {
                    probs = &probs / &sum;
                }

                let sampled_action_idx: Vec<Vec<i64>> = tch::no_grad(|| probs.multinomial(1, true))
                    .try_into()
                    .unwrap();
                let action_idx = sampled_action_idx[0][0];

                let log_prob = log_probs.double_value(&[0, action_idx]) as f32;
                let value = value_tensor.double_value(&[0]) as f32;

                let action: Action = env.action_space.get_action(action_idx as usize);
                env.step(&action);

                is_done = action == Action::Done;
                let mut reward = env
                    .pool
                    .calc_reward(&mut runtime, &env.state.machine, is_done);
                if reward.is_nan() || reward.is_infinite() {
                    reward = -1.0;
                }

                ep_reward += reward;

                // Build shifted target tokens locally to save in buffer
                let mut target_tokens = actions.clone();
                target_tokens.insert(0, 0); // SOS
                target_tokens.truncate(seq_len as usize);
                while target_tokens.len() < seq_len as usize {
                    target_tokens.push(0);
                }
                let shifted_tgt = Tensor::from_slice(&target_tokens).to(device); // (seq_len)

                buffer.states.push((
                    shifted_tgt,
                    alpha_matrix.squeeze_dim(0),
                    actions.len() as i64,
                    mask,
                )); // alpha_matrix is (1755, 5)
                buffer.actions.push(action_idx);
                buffer.log_probs.push(log_prob);
                buffer.values.push(value);
                buffer.rewards.push(reward as f32);
                buffer.dones.push(is_done);

                actions.push(action_idx);
                step_count += 1;

                if is_done {
                    // insert to pool
                    let callgraph = env.state.machine.callgraph.clone();
                    env.pool.insert(&mut runtime, callgraph);
                    break;
                }
            }

            if !is_done {
                let len = buffer.rewards.len();
                buffer.rewards[len - 1] -= unfinished_penaly as f32;
                ep_reward -= unfinished_penaly;

                let mask: Tensor = env.get_valid_action_mask(&device);
                let (_, _, value_tensor) =
                    model.forward(&env.state, &mut runtime, &mask, &device, &actions);
                last_value_for_gae = value_tensor.double_value(&[0]) as f32;
                if last_value_for_gae.is_nan() {
                    last_value_for_gae = 0.0;
                }
            } else {
                last_value_for_gae = 0.0;
            }

            ep_rewards.push(ep_reward);
            ep_lengths.push(step_count);
            ep_exprs.push(
                env.state
                    .machine
                    .callgraph
                    .format_node(env.state.machine.callgraph.root),
            );
        }

        let avg_reward: f64 = ep_rewards.iter().sum::<f64>() / ep_rewards.len() as f64;
        let avg_length: f64 = ep_lengths.iter().sum::<usize>() as f64 / ep_lengths.len() as f64;
        println!(
            "Avg Reward: {:.4} | Avg Length: {:.1} | Pool Size: {}",
            avg_reward,
            avg_length,
            env.pool.len()
        );

        let (mut returns, mut advantages) = buffer.calc_gae(last_value_for_gae, 0.99, 0.95);
        for i in 0..returns.len() {
            if returns[i].is_nan() || returns[i].is_infinite() {
                returns[i] = 0.0;
            }
            if advantages[i].is_nan() || advantages[i].is_infinite() {
                advantages[i] = 0.0;
            }
        }
        let adv_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let adv_std = (advantages
            .iter()
            .map(|x| (x - adv_mean).powi(2))
            .sum::<f32>()
            / advantages.len() as f32)
            .sqrt();
        for x in advantages.iter_mut() {
            *x = (*x - adv_mean) / (adv_std + 1e-8);
        }

        let _old_log_probs = Tensor::from_slice(&buffer.log_probs).to(device);
        let _old_actions = Tensor::from_slice(&buffer.actions).to(device);
        let _advantages_t = Tensor::from_slice(&advantages).to(device);
        let _returns_t = Tensor::from_slice(&returns).to(device);

        let mut indices: Vec<usize> = (0..buffer.states.len()).collect();
        let mut rng = rand::thread_rng();

        for _epoch in 0..epochs {
            indices.shuffle(&mut rng);
            let mut total_policy_loss = 0.0;
            let mut total_value_loss = 0.0;
            let mut total_entropy = 0.0;

            for start_idx in (0..indices.len()).step_by(batch_size) {
                let end_idx = std::cmp::min(start_idx + batch_size, indices.len());
                let batch_indices = &indices[start_idx..end_idx];

                let mut shifted_tgts = Vec::new();
                let mut data_pts = Vec::new();
                let mut step_idxs = Vec::new();
                let mut masks = Vec::new();

                let mut batch_old_log_probs = Vec::new();
                let mut batch_advantages = Vec::new();
                let mut batch_returns = Vec::new();
                let mut batch_old_actions = Vec::new();

                for &idx in batch_indices {
                    let (shifted_tgt, data_pt, step_idx, mask) = &buffer.states[idx];
                    shifted_tgts.push(shifted_tgt.shallow_clone());
                    data_pts.push(data_pt.shallow_clone());
                    step_idxs.push(*step_idx);
                    masks.push(mask.shallow_clone());

                    batch_old_log_probs.push(buffer.log_probs[idx]);
                    batch_advantages.push(advantages[idx]);
                    batch_returns.push(returns[idx]);
                    batch_old_actions.push(buffer.actions[idx] as i64);
                }

                let shifted_tgts_t = Tensor::stack(&shifted_tgts, 0).to(device); // (b, seq_len)
                let data_pts_t = Tensor::stack(&data_pts, 0).to(device); // (b, 1755, 5)
                let masks_t = Tensor::stack(&masks, 0).to(device); // (b, vocab)

                let b_old_log_probs = Tensor::from_slice(&batch_old_log_probs).to(device);
                let b_advantages = Tensor::from_slice(&batch_advantages).to(device);
                let b_returns = Tensor::from_slice(&batch_returns).to(device);
                let b_old_actions = Tensor::from_slice(&batch_old_actions).to(device);
                let step_idxs_t = Tensor::from_slice(&step_idxs).to(device);

                let (new_logits, new_values) = model.decoder.forward(&shifted_tgts_t, &data_pts_t);

                let b_size = batch_indices.len() as i64;
                let masked_logits = new_logits
                    .gather(
                        1,
                        &step_idxs_t
                            .unsqueeze(1)
                            .unsqueeze(2)
                            .expand([b_size, 1, new_logits.size()[2]], false),
                        false,
                    )
                    .squeeze_dim(1);
                let masked_logits =
                    masked_logits.masked_fill(&masks_t.logical_not(), std::f64::NEG_INFINITY);

                let new_log_probs = masked_logits.log_softmax(-1, tch::Kind::Float);
                let new_probs = new_log_probs.exp();
                let sums = new_probs.sum_dim_intlist(Some(&[-1][..]), true, tch::Kind::Float);
                let safe_sums = sums.clamp_min(1e-8);
                let new_probs = (&new_probs / &safe_sums).where_self(
                    &sums.gt(0.0),
                    &(Tensor::ones_like(&new_probs) / (new_probs.size()[1] as f64)),
                );

                // Calculate log prob of old action
                let new_log_prob = new_probs
                    .gather(1, &b_old_actions.unsqueeze(1), false)
                    .squeeze_dim(1)
                    .clamp_min(1e-8)
                    .log();

                let safe_log_probs = new_log_probs.clamp_min(-1e8);
                let entropy = -(new_probs.shallow_clone() * &safe_log_probs)
                    .sum_dim_intlist(Some(&[-1][..]), false, tch::Kind::Float)
                    .mean(tch::Kind::Float);

                let ratio = (new_log_prob - b_old_log_probs).exp();
                let surr1 = &ratio * &b_advantages;
                let surr2 = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * &b_advantages;
                let policy_loss = -surr1.min_other(&surr2).mean(tch::Kind::Float);

                let new_val_extracted = new_values
                    .gather(1, &step_idxs_t.unsqueeze(1), false)
                    .squeeze_dim(1);
                let value_loss = new_val_extracted.mse_loss(&b_returns, tch::Reduction::Mean);

                let policy_val = policy_loss.double_value(&[]);
                let value_val = value_loss.double_value(&[]);
                let entropy_val = entropy.double_value(&[]);

                total_policy_loss += policy_val * (batch_indices.len() as f64);
                total_value_loss += value_val * (batch_indices.len() as f64);
                total_entropy += entropy_val * (batch_indices.len() as f64);

                let loss = policy_loss + value_loss * c1 - entropy * c2;
                let loss_val = loss.double_value(&[]);

                if loss_val.is_nan()
                    || loss_val.is_infinite()
                    || policy_val.is_nan()
                    || policy_val.is_infinite()
                    || value_val.is_nan()
                    || value_val.is_infinite()
                    || entropy_val.is_nan()
                    || entropy_val.is_infinite()
                {
                    println!(
                        "WARNING: Loss is {}! (Policy: {}, Value: {}, Entropy: {}) Skipping backward.",
                        loss_val, policy_val, value_val, entropy_val
                    );
                    println!("Diagnostic - Batch Returns: {:?}", batch_returns);
                    println!("Diagnostic - Batch Advantages: {:?}", batch_advantages);
                    println!(
                        "Diagnostic - Batch Old Actions (Indices): {:?}",
                        batch_old_actions
                    );
                    continue;
                }

                opt.backward_step_clip(&loss, 0.5);
            }
            let n_states = buffer.states.len() as f64;
            println!(
                "Epoch {}/{} - Policy Loss: {:.4}, Value Loss: {:.4}, Entropy: {:.4}",
                _epoch + 1,
                epochs,
                total_policy_loss / n_states,
                total_value_loss / n_states,
                total_entropy / n_states
            );
        }
    }
    env.pool
}
