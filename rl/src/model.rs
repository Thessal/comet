use runtime::runtime::Runtime;
use stdlib::types::Signal;
use tch::{
    Device,
    Kind::Float,
    Tensor,
    nn::{self, LSTMState, Module, RNN},
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
    ) -> (Tensor, Tensor) {
        // state_embedding, logits
        unimplemented!()
    }
}

pub struct RandomModel {
    pub action_space: ActionSpace,
}

impl RandomModel {
    pub fn new(action_space: ActionSpace) -> Self {
        Self { action_space }
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
    ) -> (Tensor, Tensor) {
        // (state_embedding, action_logits)}
        let logits = tch::Tensor::ones(
            [1, self.action_space.size() as i64],
            (tch::Kind::Float, *device),
        );
        let dummy_emb = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let masked_logits =
            logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);
        (dummy_emb, masked_logits)
    }
}

pub struct TransformerLayer {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    ff1: nn::Linear,
    ff2: nn::Linear,
}

impl TransformerLayer {
    pub fn new(vs: &nn::Path, embed_dim: i64) -> Self {
        let q_proj = nn::linear(vs, embed_dim, embed_dim, Default::default());
        let k_proj = nn::linear(vs, embed_dim, embed_dim, Default::default());
        let v_proj = nn::linear(vs, embed_dim, embed_dim, Default::default());
        let out_proj = nn::linear(vs, embed_dim, embed_dim, Default::default());
        let norm1 = nn::layer_norm(vs, vec![embed_dim], Default::default());
        let norm2 = nn::layer_norm(vs, vec![embed_dim], Default::default());
        let ff1 = nn::linear(vs, embed_dim, embed_dim * 4, Default::default());
        let ff2 = nn::linear(vs, embed_dim * 4, embed_dim, Default::default());
        Self { q_proj, k_proj, v_proj, out_proj, norm1, norm2, ff1, ff2 }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);
        
        let d_k = q.size()[q.size().len() - 1] as f64;
        let scores = q.matmul(&k.transpose(-2, -1)) / d_k.sqrt();
        let attn = scores.softmax(-1, Float).matmul(&v);
        
        let x = self.norm1.forward(&(x + self.out_proj.forward(&attn)));
        
        let ff_out = self.ff2.forward(&self.ff1.forward(&x).relu());
        self.norm2.forward(&(x + ff_out))
    }
}

pub struct AgentModel {
    pub action_space: ActionSpace,
    data_lstm: nn::LSTM,
    lstm_hidden: LSTMState,
    transformer: TransformerLayer,
    actor_proj: nn::Linear,
    critic_proj: nn::Linear,
    embed_dim: i64,
}

impl AgentModel {
    pub fn new(vs: &nn::Path, action_space: ActionSpace, embed_dim: i64) -> Self {
        let data_lstm = nn::lstm(vs, crate::embed::EMBEDDING_SIZE as i64, embed_dim, Default::default());
        let lstm_hidden = data_lstm.zero_state(1);
        let transformer = TransformerLayer::new(vs, embed_dim);
        let actor_proj = nn::linear(vs, embed_dim, action_space.size() as i64, Default::default());
        let critic_proj = nn::linear(vs, embed_dim, 1, Default::default());
        Self {
            action_space,
            data_lstm,
            lstm_hidden,
            transformer,
            actor_proj,
            critic_proj,
            embed_dim,
        }
    }
    
    pub fn calculate_policy_gradient_loss(
        &self,
        log_probs: &Tensor,
        advantages: &Tensor,
        entropy: Option<&Tensor>,
        entropy_coef: f64,
    ) -> Tensor {
        // REINFORCE / Policy Gradient loss = -mean(log_prob * advantage)
        let mut loss = -(log_probs * advantages).mean(tch::Kind::Float);
        
        // Add entropy regularization if provided to encourage exploration
        if let Some(ent) = entropy {
            loss = loss - ent.mean(tch::Kind::Float) * entropy_coef;
        }
        loss
    }
    
    pub fn calculate_value_loss(
        &self,
        values: &Tensor,
        returns: &Tensor,
    ) -> Tensor {
        // Critic loss = MSE(values, returns)
        values.mse_loss(returns, tch::Reduction::Mean)
    }
}

impl Model for AgentModel {
    fn reset(&self) {
        todo!()
    }
    
    fn forward(
        &mut self,
        state: &SearchState,
        runtime: &mut Runtime,
        masks: &Tensor,
        device: &Device,
    ) -> (Tensor, Tensor) {
        let (stack, callgraph) = state.machine.get_stack();
        let mut data: Vec<Option<Tensor>> = Vec::new();
        
        {
            for (_signal_decl, addr) in stack.iter() {
                let signal = runtime.lookup_or_run(callgraph, *addr);
                match signal {
                    Signal::DataFrame(Some(df)) => data.push(Some(df.shallow_clone())),
                    _ => data.push(None),
                }
            }
        }

        let mut data_embeddings: Vec<Tensor> = Vec::new();
        
        for df_opt in data {
            if let Some(_df) = df_opt {
                let df_tensor = Tensor::zeros([1, crate::embed::EMBEDDING_SIZE as i64], (Float, *device));
                self.lstm_hidden = self.data_lstm.step(&df_tensor, &self.lstm_hidden);
                data_embeddings.push(self.lstm_hidden.h().view([1, self.embed_dim]));
            } else {
                let dummy = Tensor::zeros([1, self.embed_dim], (Float, *device));
                data_embeddings.push(dummy);
            }
        }
        
        let seq_embedding = if data_embeddings.is_empty() {
            Tensor::zeros([1, 1, self.embed_dim], (Float, *device))
        } else {
            Tensor::stack(&data_embeddings, 1)
        };

        let transformer_out = self.transformer.forward(&seq_embedding);
        let state_embedding = transformer_out.mean_dim(Some([1].as_slice()), false, Float);
        
        let logits = self.actor_proj.forward(&state_embedding);
        let masked_logits = logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);
        let _value = self.critic_proj.forward(&state_embedding);
        
        (state_embedding, masked_logits)
    }
}

// // #[cfg(test)]
// // mod tests {
// //     use super::*;
// //     use crate::action::ActionSpace;
// //     use parser::behavior::test_make_behavior;
// //     use tch::{Device, Kind};

// //     #[test]
// //     fn test_rnn_inference_with_masking() {
// //         let behavior = test_make_behavior();
// //         let action_space: ActionSpace = (&behavior).into();
// //         let action_vocab_size = action_space.size();

// //         // 1. Initialize the RNN model
// //         let vs = nn::VarStore::new(Device::Cpu);
// //         let config = ModelSize::Small.get_config(action_vocab_size);
// //         let model = config.init(&vs.root());

// //         // 2. Create dummy state sequence [batch_size=1, state_size=crate::env::EMBEDDING_SIZE]
// //         let states = Tensor::from_slice(&[1_i64; crate::env::EMBEDDING_SIZE])
// //             .view([1, crate::env::EMBEDDING_SIZE as i64]);

// //         let mut mask = vec![false; action_vocab_size];
// //         for i in (0..action_vocab_size).step_by(2) {
// //             mask[i] = true;
// //         }
// //         let available_actions = Tensor::from_slice(&mask).view([1, action_vocab_size as i64]);

// //         // 3. Run Forward Pass
// //         let lstm_state: Option<LSTMState> = None;
// //         let (logits_not_masked, _lstm_state) = model.forward(states, &lstm_state); // not masked
// //         assert_eq!(logits_not_masked.size()[1], action_vocab_size as i64);
// //         let logits =
// //             logits_not_masked.masked_fill(&available_actions.logical_not(), f64::NEG_INFINITY);

// //         assert_eq!(logits.size(), vec![1, action_vocab_size as i64]);

// //         // 6. Compute probabilities
// //         let probabilities = logits.softmax(-1, Kind::Float);

// //         // 7. Verify boundaries
// //         let probs_array: Vec<f32> = probabilities.squeeze_dim(0).try_into().unwrap();
// //         for i in (0..action_vocab_size).step_by(2) {
// //             assert!(probs_array[i] > 0.0);
// //             assert_eq!(probs_array[i + 1], 0.0);
// //         }

// //         let valid_sum: f32 = probs_array.iter().sum();
// //         assert!((valid_sum - 1.0).abs() < 1e-5);
// //     }
// // }
