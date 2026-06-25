use tch::{Device, Kind, Tensor, nn, nn::Module};

pub struct AlphaConvEncoder {
    conv1d: nn::Conv1D,
    proj: nn::Linear,
    d_model: i64,
}

impl AlphaConvEncoder {
    pub fn new(vs: &nn::Path, hidden_dim: i64, d_model: i64, kernel_size: i64) -> Self {
        let conv_config = nn::ConvConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            ..Default::default()
        };
        let conv1d = nn::conv1d(vs, 1, hidden_dim, kernel_size, conv_config);
        let proj = nn::linear(vs, hidden_dim, d_model, Default::default());
        Self {
            conv1d,
            proj,
            d_model,
        }
    }

    pub fn forward(&self, alpha_matrix: &Tensor) -> Tensor {
        // alpha_matrix: (batch_size, time_steps, num_instruments)
        let mut alpha = alpha_matrix.shallow_clone();
        if alpha.dim() == 2 {
            alpha = alpha.unsqueeze(0);
        }
        let size = alpha.size();
        let batch_size = size[0];
        let time_steps = size[1];
        let num_instruments = size[2];

        let x = alpha.transpose(1, 2).contiguous();
        let x = x.view([batch_size * num_instruments, 1, time_steps]);
        let x = x.nan_to_num(0.0, 0.0, 0.0);

        let x = x.apply(&self.conv1d).relu();
        let x = x.max_dim(-1, false).0;
        let x = x.view([batch_size, num_instruments, -1]);
        let x = x.mean_dim(Some([1].as_slice()), false, Kind::Float);

        self.proj.forward(&x)
    }
}

pub struct TransformerDecoderLayer {
    self_attn_q: nn::Linear,
    self_attn_k: nn::Linear,
    self_attn_v: nn::Linear,
    self_attn_out: nn::Linear,

    cross_attn_q: nn::Linear,
    cross_attn_k: nn::Linear,
    cross_attn_v: nn::Linear,
    cross_attn_out: nn::Linear,

    ln1: nn::LayerNorm,
    ln2: nn::LayerNorm,
    ln3: nn::LayerNorm,

    mlp1: nn::Linear,
    mlp2: nn::Linear,

    num_heads: i64,
    head_dim: i64,
}

impl TransformerDecoderLayer {
    pub fn new(vs: &nn::Path, embed_dim: i64, num_heads: i64, dim_feedforward: i64) -> Self {
        let head_dim = embed_dim / num_heads;
        Self {
            self_attn_q: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            self_attn_k: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            self_attn_v: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            self_attn_out: nn::linear(vs, embed_dim, embed_dim, Default::default()),

            cross_attn_q: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            cross_attn_k: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            cross_attn_v: nn::linear(vs, embed_dim, embed_dim, Default::default()),
            cross_attn_out: nn::linear(vs, embed_dim, embed_dim, Default::default()),

            ln1: nn::layer_norm(vs, vec![embed_dim], Default::default()),
            ln2: nn::layer_norm(vs, vec![embed_dim], Default::default()),
            ln3: nn::layer_norm(vs, vec![embed_dim], Default::default()),

            mlp1: nn::linear(vs, embed_dim, dim_feedforward, Default::default()),
            mlp2: nn::linear(vs, dim_feedforward, embed_dim, Default::default()),

            num_heads,
            head_dim,
        }
    }

    pub fn forward(&self, tgt: &Tensor, memory: &Tensor, tgt_mask: &Tensor) -> Tensor {
        let size = tgt.size();
        let b = size[0];
        let s = size[1];
        let e = size[2];

        // 1. Self Attention
        let q = self
            .self_attn_q
            .forward(tgt)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = self
            .self_attn_k
            .forward(tgt)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = self
            .self_attn_v
            .forward(tgt)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let scores = scores + tgt_mask.unsqueeze(0).unsqueeze(0); // Add causal mask
        let attn_weights = scores.softmax(-1, Kind::Float);

        let context = attn_weights
            .matmul(&v)
            .transpose(1, 2)
            .contiguous()
            .view([b, s, e]);
        let self_attn_out = self.self_attn_out.forward(&context);

        let tgt = (tgt + self_attn_out).apply(&self.ln1);

        // 2. Cross Attention
        let memory_size = memory.size();
        let mem_s = memory_size[1];

        let q_cross = self
            .cross_attn_q
            .forward(&tgt)
            .view([b, s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k_cross = self
            .cross_attn_k
            .forward(memory)
            .view([b, mem_s, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v_cross = self
            .cross_attn_v
            .forward(memory)
            .view([b, mem_s, self.num_heads, self.head_dim])
            .transpose(1, 2);

        let scores_cross =
            q_cross.matmul(&k_cross.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let attn_weights_cross = scores_cross.softmax(-1, Kind::Float);

        let context_cross = attn_weights_cross
            .matmul(&v_cross)
            .transpose(1, 2)
            .contiguous()
            .view([b, s, e]);
        let cross_attn_out = self.cross_attn_out.forward(&context_cross);

        let tgt = (tgt + cross_attn_out).apply(&self.ln2);

        // 3. MLP
        let mlp_out = self.mlp2.forward(&self.mlp1.forward(&tgt).relu());

        (tgt + mlp_out).apply(&self.ln3)
    }
}

pub struct SRDecoderModel {
    pub d_model: i64,
    alpha_context_encoder: AlphaConvEncoder,
    target_token_emb: nn::Embedding,
    pos_embedding: Tensor,
    layers: Vec<TransformerDecoderLayer>,
    output_linear: nn::Linear,
    value_linear: nn::Linear,
}

impl SRDecoderModel {
    pub fn new(
        vs: &nn::Path,
        vocab_size: i64,
        d_model: i64,
        nhead: i64,
        nhid: i64,
        nlayers: i64,
    ) -> Self {
        let alpha_context_encoder =
            AlphaConvEncoder::new(&vs.sub("alpha_context_encoder"), 64, d_model, 5);
        let target_token_emb = nn::embedding(
            &vs.sub("target_token_emb"),
            vocab_size,
            d_model,
            Default::default(),
        );
        let pos_embedding = vs.var("pos_embedding", &[1, 500, d_model], nn::Init::Const(0.0));

        let mut layers = Vec::new();
        let vs_layers = vs.sub("transformer_decoder").sub("layers");
        for i in 0..nlayers {
            layers.push(TransformerDecoderLayer::new(
                &vs_layers.sub(format!("{}", i)),
                d_model,
                nhead,
                nhid,
            ));
        }

        let output_linear = nn::linear(
            &vs.sub("output_linear"),
            d_model,
            vocab_size,
            Default::default(),
        );
        let value_linear = nn::linear(&vs.sub("value_linear"), d_model, 1, Default::default());

        Self {
            d_model,
            alpha_context_encoder,
            target_token_emb,
            pos_embedding,
            layers,
            output_linear,
            value_linear,
        }
    }

    pub fn forward(
        &self,
        shifted_target_tokens: &Tensor,
        alpha_matrix: &Tensor,
    ) -> (Tensor, Tensor) {
        let size = shifted_target_tokens.size();
        let _batch_size = size[0];
        let seq_len = size[1];
        let device = shifted_target_tokens.device();

        let memory_context = self
            .alpha_context_encoder
            .forward(alpha_matrix)
            .unsqueeze(1);

        let tgt_vectors =
            self.target_token_emb.forward(shifted_target_tokens) * (self.d_model as f64).sqrt();
        let pos_emb_slice = self.pos_embedding.slice(1, 0, seq_len, 1);
        let mut tgt_vectors = tgt_vectors + pos_emb_slice;

        // Causal mask: -inf on upper triangle, 0 on lower
        let causal_mask = Tensor::ones([seq_len, seq_len], (Kind::Float, device))
            .triu(1)
            .masked_fill(
                &Tensor::ones([seq_len, seq_len], (Kind::Bool, device)).triu(1),
                std::f64::NEG_INFINITY,
            );

        for layer in &self.layers {
            tgt_vectors = layer.forward(&tgt_vectors, &memory_context, &causal_mask);
        }

        let logits = self.output_linear.forward(&tgt_vectors);
        let values = self.value_linear.forward(&tgt_vectors).squeeze_dim(-1);

        (logits, values)
    }
}
