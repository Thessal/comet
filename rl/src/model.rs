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
    ) -> (Tensor, Tensor, Tensor) {
        // state_embedding, logits, value
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
    ) -> (Tensor, Tensor, Tensor) {
        // (state_embedding, action_logits, value)}
        let logits = tch::Tensor::ones(
            [1, self.action_space.size() as i64],
            (tch::Kind::Float, *device),
        );
        let dummy_emb = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let dummy_val = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let masked_logits =
            logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);
        (dummy_emb, masked_logits, dummy_val)
    }
}

pub struct TransformerLayer {
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    ln1: nn::LayerNorm,
    mlp1: nn::Linear,
    mlp2: nn::Linear,
    ln2: nn::LayerNorm,
    num_heads: i64,
    head_dim: i64,
}

impl TransformerLayer {
    pub fn new(vs: &nn::Path, embed_dim: i64) -> Self {
        let num_heads = 4;
        let head_dim = embed_dim / num_heads;
        Self {
            q_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            k_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            v_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            out_proj: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            ln1: nn::layer_norm(vs, vec![embed_dim], Default::default()),
            mlp1: nn::linear(vs, embed_dim, 4 * embed_dim, Default::default()),
            mlp2: nn::linear(vs, 4 * embed_dim, embed_dim, Default::default()),
            ln2: nn::layer_norm(vs, vec![embed_dim], Default::default()),
            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Tensor {
        let size = x.size();
        let b = size[0];
        let s = size[1];
        let e = size[2];

        let ln_x = x.apply(&self.ln1);

        let q = self
            .q_proj
            .forward(&ln_x)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = self
            .k_proj
            .forward(&ln_x)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = self
            .v_proj
            .forward(&ln_x)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let scores = scores.masked_fill(mask, std::f64::NEG_INFINITY);
        let attn_weights = scores.softmax(-1, tch::Kind::Float);

        let context = attn_weights
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([b, s, e]);
        let mha_out = self.out_proj.forward(&context);

        let x1 = x + &mha_out;
        let ln_x2 = x1.apply(&self.ln2);
        let mlp_out = self.mlp2.forward(&self.mlp1.forward(&ln_x2).relu());

        x1 + mlp_out
    }
}

pub struct AgentModel {
    pub action_space: ActionSpace,
    transformer_layers: Vec<TransformerLayer>,
    actor_proj: nn::Linear,
    critic_proj: nn::Linear,
    node_embedding: nn::Embedding,
    data_proj: nn::Linear,
    embed_dim: i64,
}

impl AgentModel {
    pub fn new(vs: &nn::Path, action_space: ActionSpace, embed_dim: i64) -> Self {
        let mut transformer_layers = Vec::new();
        let depth = 4;
        for i in 0..depth {
            transformer_layers.push(TransformerLayer::new(
                &vs.sub(format!("layer_{}", i)),
                embed_dim,
            ));
        }
        let actor_proj = nn::linear(
            vs,
            embed_dim,
            action_space.size() as i64,
            Default::default(),
        );
        let critic_proj = nn::linear(vs, embed_dim, 1, Default::default());
        let node_embedding = nn::embedding(vs, 1000, embed_dim, Default::default());
        let data_proj = nn::linear(vs, stdlib::types::SIZE[1], embed_dim, Default::default());

        Self {
            action_space,
            transformer_layers,
            actor_proj,
            critic_proj,
            node_embedding,
            data_proj,
            embed_dim,
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
        // PPO clipped surrogate objective
        let ratio = (log_probs - old_log_probs).exp();
        let loss1 = &ratio * advantages;
        let loss2 = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages;

        let mut policy_loss = -loss1.min_other(&loss2).mean(tch::Kind::Float);

        // Add entropy regularization if provided to encourage exploration
        if let Some(ent) = entropy {
            policy_loss = policy_loss - ent.mean(tch::Kind::Float) * entropy_coef;
        }
        policy_loss
    }

    pub fn calculate_value_loss(&self, values: &Tensor, returns: &Tensor) -> Tensor {
        // Critic loss = MSE(values, returns)
        values.mse_loss(returns, tch::Reduction::Mean)
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
    ) -> (Tensor, Tensor, Tensor) {
        let (stack, callgraph) = state.machine.get_stack();
        let mut data_tensors = Vec::new();

        for (_signal_decl, addr) in stack.iter() {
            let signal = runtime.lookup_or_run(callgraph, *addr);
            if let stdlib::types::Signal::DataFrame(Some(df)) = signal {
                let df_mean = df.mean_dim(Some([0].as_slice()), false, tch::Kind::Float);
                data_tensors.push(df_mean);
            }
        }

        let stack_data_emb = if data_tensors.is_empty() {
            Tensor::zeros([1, self.embed_dim], (tch::Kind::Float, *device))
        } else {
            let stacked = Tensor::stack(&data_tensors, 0).to(*device);
            let proj = self.data_proj.forward(&stacked);
            proj.mean_dim(Some([0].as_slice()), false, tch::Kind::Float)
                .unsqueeze(0)
        };
        let stack_data_emb_seq = stack_data_emb.unsqueeze(1);

        let mut node_indices = Vec::new();
        for node in &callgraph.nodes {
            let id = match &node.node_type {
                parser::ast::NodeType::Operator(_) => 1,
                parser::ast::NodeType::Literal(_) => 2,
                parser::ast::NodeType::Behavior(_) => 3,
            };
            node_indices.push(id);
        }

        let input_seq = if node_indices.is_empty() {
            stack_data_emb_seq
        } else {
            let node_indices_t = Tensor::from_slice(&node_indices)
                .to_kind(tch::Kind::Int64)
                .to(*device);
            let seq_emb = self.node_embedding.forward(&node_indices_t).unsqueeze(0);
            Tensor::concat(&[stack_data_emb_seq, seq_emb], 1)
        };

        let seq_len = input_seq.size()[1];
        let ones = Tensor::ones([seq_len, seq_len], (tch::Kind::Bool, *device));
        let mask = ones.triu(1);

        let mut x = input_seq;
        for layer in &self.transformer_layers {
            x = layer.forward(&x, &mask);
        }

        let state_embedding = x.select(1, -1);

        let logits = self.actor_proj.forward(&state_embedding);
        let value = self.critic_proj.forward(&state_embedding);

        let masked_logits =
            logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);

        (state_embedding, masked_logits, value)
    }
}
