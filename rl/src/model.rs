use runtime::runtime::Runtime;
use stdlib::types;
use tch::{
    Device, Tensor,
    nn::{self, Module},
};

use crate::{action::ActionSpace, state::SearchState};

pub trait Model {
    fn reset(&self) {
        // resets internal state of time series models
        unimplemented!()
    }
    fn forward(
        &mut self,
        _state: &SearchState,
        _runtime: &mut Runtime,
        _masks: &Tensor,
        _device: &Device,
        _actions: &[i64],
    ) -> (Tensor, Tensor, Tensor) {
        // state_embedding, logits, value
        unimplemented!()
    }
}

pub struct RandomModel {
    pub action_space: ActionSpace,

    // Note that this is only approximation because some operators are not available depending on data type in stack.
    // For accurate calculation, transition probability matrix need to be calculated.
    // Or, hierarchical sampling need to be used, which is too complex to be used as a benchmark.
    // See result/transition.ipynb
    log_w_intro: f64, //probability of introducing new parameter / variable
    log_w_done: f64,  //conditional probability of stop when stop action is possible.
    intro_idxs: Vec<usize>,
    done_idxs: Vec<usize>,
}

impl RandomModel {
    pub fn new(action_space: ActionSpace, introduce_prob: f64, stop_prob: f64) -> Self {
        let mut intro_idxs: Vec<usize> = vec![];
        let mut done_idxs: Vec<usize> = vec![];
        for i in 0..action_space.size() {
            match action_space.get_action(i) {
                crate::action::Action::Reduce(_) => {} // operator_actions+=1,
                crate::action::Action::Done => done_idxs.push(i),
                _ => intro_idxs.push(i),
            }
        }
        // stop_prob = w / (other actions + 1) ~= w / (introduce_actions + 1)
        let log_w_done = (stop_prob * (intro_idxs.len() + 1) as f64).ln();
        // intro_prob = w * (intro_actions) / (all available_actions) ~= w * (intro_actions) / (size)
        let log_w_intro = (introduce_prob * (action_space.size() as f64)).ln();

        Self {
            action_space,
            log_w_intro,
            log_w_done,
            intro_idxs,
            done_idxs,
        }
    }
}

impl Model for RandomModel {
    fn reset(&self) {
        todo!() // Initilaize lstm_hidden
    }
    fn forward(
        &mut self,
        _state: &SearchState,
        _runtime: &mut Runtime,
        masks: &Tensor,
        device: &Device,
        _actions: &[i64],
    ) -> (Tensor, Tensor, Tensor) {
        // (state_embedding, action_logits, value)}

        let mut logits = vec![0f64; self.action_space.size()];
        for i in &self.intro_idxs {
            logits[*i] += self.log_w_intro
        }
        for i in &self.done_idxs {
            logits[*i] += self.log_w_done
        }
        let logits = tch::Tensor::from_slice(&logits)
            .to_kind(tch::Kind::Float)
            .to_device(*device)
            .unsqueeze(0);

        let dummy_emb = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let dummy_val = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let masked_logits =
            logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);

        (dummy_emb, masked_logits, dummy_val)
    }
}

// pub struct TransformerLayer {
//     q_proj: nn::Linear,
//     k_proj: nn::Linear,
//     v_proj: nn::Linear,
//     out_proj: nn::Linear,
//     ln1: nn::LayerNorm,
//     mlp1: nn::Linear,
//     mlp2: nn::Linear,
//     ln2: nn::LayerNorm,
//     num_heads: i64,
//     head_dim: i64,
// }

// impl TransformerLayer {
//     pub fn new(vs: &nn::Path, embed_dim: i64) -> Self {
//         let num_heads = 4;
//         let head_dim = embed_dim / num_heads;
//         Self {
//             q_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
//             k_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
//             v_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
//             out_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
//             ln1: nn::layer_norm(vs, vec![embed_dim], Default::default()),
//             mlp1: nn::linear(vs, embed_dim, 4 * embed_dim, Default::default()),
//             mlp2: nn::linear(vs, 4 * embed_dim, embed_dim, Default::default()),
//             ln2: nn::layer_norm(vs, vec![embed_dim], Default::default()),
//             num_heads,
//             head_dim,
//         }
//     }

//     pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Tensor {
//         let size = x.size();
//         let b = size[0];
//         let s = size[1];
//         let e = size[2];

//         let ln_x = x.apply(&self.ln1);

//         let q = self
//             .q_proj
//             .forward(&ln_x)
//             .view([b, s, self.num_heads, self.head_dim])
//             .transpose(1, 2);
//         let k = self
//             .k_proj
//             .forward(&ln_x)
//             .view([b, s, self.num_heads, self.head_dim])
//             .transpose(1, 2);
//         let v = self
//             .v_proj
//             .forward(&ln_x)
//             .view([b, s, self.num_heads, self.head_dim])
//             .transpose(1, 2);

//         let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
//         let scores = scores.masked_fill(mask, std::f64::NEG_INFINITY);
//         let attn_weights = scores.softmax(-1, tch::Kind::Float);

//         let context = attn_weights
//             .matmul(&v)
//             .transpose(1, 2)
//             .contiguous()
//             .view([b, s, e]);
//         let mha_out = self.out_proj.forward(&context);

//         let x1 = x + &mha_out;
//         let ln_x2 = x1.apply(&self.ln2);
//         let mlp_out = self.mlp2.forward(&self.mlp1.forward(&ln_x2).relu());

//         x1 + mlp_out
//     }
// }

pub struct AgentModel {
    pub action_space: ActionSpace,
    pub decoder: crate::model_transformer::SRDecoderModel,
    pub value_net: nn::Sequential,
    seq_len: i64,
}

impl AgentModel {
    pub fn new(vs: &nn::Path, action_space: ActionSpace, d_model: i64) -> Self {
        let vocab_size = action_space.size() as i64;
        let decoder =
            crate::model_transformer::SRDecoderModel::new(vs, vocab_size, d_model, 8, 512, 4);

        let vs_v = vs.sub("value_net");
        let value_net = nn::seq()
            .add(nn::linear(&vs_v.sub("l1"), 5, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&vs_v.sub("l2"), 64, 64, Default::default()))
            .add_fn(|x| x.relu())
            .add(nn::linear(&vs_v.sub("l3"), 64, 2, Default::default()));

        Self {
            action_space,
            decoder,
            value_net,
            seq_len: 50,
        }
    }

    pub fn calculate_ppo_loss(
        &self,
        log_probs: &Tensor,
        old_log_probs: &Tensor,
        advantages: &Tensor,
        entropy: Option<&Tensor>,
        entropy_coef: f64,
        clip_coef: f64,
    ) -> Tensor {
        let ratio = (log_probs - old_log_probs).exp();
        let loss1 = &ratio * advantages;
        let loss2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;

        let mut policy_loss = -loss1.min_other(&loss2).mean(tch::Kind::Float);

        if let Some(ent) = entropy {
            policy_loss = policy_loss - ent.mean(tch::Kind::Float) * entropy_coef;
        }
        policy_loss
    }

    pub fn calculate_value_loss(&self, values: &Tensor, returns: &Tensor) -> Tensor {
        values.mse_loss(returns, tch::Reduction::Mean)
    }

    pub fn compute_5d_embedding(
        history: &[i64],
        is_done: bool,
        action_space: &ActionSpace,
    ) -> [f32; 5] {
        let len = history.len() as f32;
        let mut num_data = 0.0;
        let mut num_ops = 0.0;

        let mut counts = std::collections::HashMap::new();
        for &act_idx in history {
            *counts.entry(act_idx).or_insert(0) += 1;
            let action = action_space.get_action(act_idx as usize);
            match action {
                crate::action::Action::ShiftString(_) | crate::action::Action::ShiftParam(_) => {
                    num_data += 1.0;
                }
                crate::action::Action::Reduce(_) => {
                    num_ops += 1.0;
                }
                _ => {}
            }
        }

        let mut entropy = 0.0;
        if len > 0.0 {
            for &count in counts.values() {
                let p = (count as f32) / len;
                entropy -= p * p.ln();
            }
        }

        // Scale roughly to [0, 1] based on max length of 30
        let scale = 30.0;
        [
            len / scale,
            num_data / scale,
            num_ops / scale,
            entropy,
            if is_done { 1.0 } else { 0.0 },
        ]
    }
}

impl Model for AgentModel {
    fn reset(&self) {}

    fn forward(
        &mut self,
        state: &SearchState,
        runtime: &mut Runtime,
        masks: &Tensor,
        device: &Device,
        actions: &[i64],
    ) -> (Tensor, Tensor, Tensor) {
        let (stack, callgraph) = state.machine.get_stack();
        let mut data_tensors = Vec::new();

        for (_signal_decl, addr) in stack.iter() {
            let signal = runtime.lookup_or_run(callgraph, *addr);
            if let stdlib::types::Signal::DataFrame(Some(df)) = signal {
                data_tensors.push(df.to_device(tch::Device::Cpu));
            }
        }

        let mut alpha_matrix = if data_tensors.is_empty() {
            Tensor::zeros(
                [types::SIZE[0], types::SIZE[1]],
                (tch::Kind::Float, *device),
            )
        } else {
            let stacked = Tensor::stack(&data_tensors, 0);
            let size = stacked.size();
            let flattened: Vec<f32> = stacked.flatten(0, -1).try_into().unwrap_or_default();
            let stacked = Tensor::from_slice(&flattened)
                .view([size[0], size[1], size[2]])
                .to(*device);
            stacked.mean_dim(Some([0].as_slice()), false, tch::Kind::Float)
        };
        alpha_matrix = alpha_matrix.nan_to_num(0.0, 0.0, 0.0);
        let alpha_matrix = alpha_matrix.unsqueeze(0); // batch_size=1

        // Build shifted target tokens
        let mut target_tokens = actions.to_vec();
        target_tokens.insert(0, 0); // SOS
        target_tokens.truncate(self.seq_len as usize);
        while target_tokens.len() < self.seq_len as usize {
            target_tokens.push(0); // pad
        }
        let shifted_target_tokens = Tensor::from_slice(&target_tokens).to(*device).unsqueeze(0);

        let logits = self.decoder.forward(&shifted_target_tokens, &alpha_matrix);

        let step_idx = std::cmp::min(actions.len(), self.seq_len as usize - 1) as i64;
        let step_logits = logits.select(1, step_idx); // (1, vocab_size)

        let emb_5d = Self::compute_5d_embedding(actions, false, &self.action_space);
        let emb_5d_t = Tensor::from_slice(&emb_5d).to(*device).unsqueeze(0);
        let value_out = self.value_net.forward(&emb_5d_t).squeeze_dim(0); // (2,)

        let mu = value_out.select(0, 0);
        let log_sigma = value_out.select(0, 1).clamp(-20.0, 2.0);
        let sigma = log_sigma.exp();
        let value: tch::Tensor = &mu - 1.96 * &sigma; // 5% quantile // 0.05 0.95 : 2-sigma

        // // Debug printing (randomly print ~1% of the time to avoid spam, or just print if needed)
        // if rand::random::<f32>() < 0.01 {
        //     let mu_val = mu.double_value(&[]);
        //     let sigma_val = sigma.double_value(&[]);
        //     println!(
        //         "Debug ValueNet -> Inputs: {:?}, mu: {:.4}, sigma: {:.4}, value: {:.4}",
        //         emb_5d,
        //         mu_val,
        //         sigma_val,
        //         value.double_value(&[])
        //     );
        // }

        let masked_logits =
            step_logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);

        (alpha_matrix, masked_logits, value)
    }
}
