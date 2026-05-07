use tch::{
    Tensor,
    nn::{self, LSTMState, Module, RNN},
};

pub enum ModelConfig {
    RnnModel(RnnModelConfig),
    // TransformerModel(TransformerModelConfig),
}

#[derive(Debug)]
pub enum Model {
    RnnModel(RnnModel),
    // TransformerModel(TransformerModel)
}

// impl Module for Model {
impl Model {
    pub fn forward(
        &self,
        input: Tensor,
        hidden: &Option<LSTMState>,
    ) -> (Tensor, Option<LSTMState>) {
        match self {
            Model::RnnModel(model) => model.forward(input, hidden),
        }
    }
}

impl ModelConfig {
    pub fn init(&self, vs: &nn::Path) -> Model {
        match self {
            ModelConfig::RnnModel(config) => Model::RnnModel(config.init(vs)),
        }
    }
}

#[derive(Debug)]
pub struct RnnModelConfig {
    pub action_vocab_size: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    pub dropout: f64,
}

impl RnnModelConfig {
    pub fn new(action_vocab_size: usize, d_model: usize, d_hidden: usize) -> Self {
        Self {
            action_vocab_size,
            d_model,
            d_hidden,
            dropout: 0.1,
        }
    }
    pub fn init(&self, vs: &nn::Path) -> RnnModel {
        let action_embedding = nn::embedding(
            vs,
            self.action_vocab_size as i64,
            self.d_model as i64,
            Default::default(),
        );

        // RNN takes the concatenation of parent and sibling embeddings
        let rnn = nn::lstm(
            vs,
            crate::env::EMBEDDING_SIZE as i64,
            self.d_hidden as i64,
            Default::default(),
        );

        let output_proj = nn::linear(
            vs,
            self.d_hidden as i64,
            self.action_vocab_size as i64,
            Default::default(),
        );

        RnnModel {
            action_embedding,
            rnn,
            output_proj,
        }
    }
}

pub enum ModelSize {
    Small,
    Base,
    Large,
}

impl ModelSize {
    pub fn get_config(&self, action_vocab_size: usize) -> RnnModelConfig {
        match self {
            ModelSize::Small => RnnModelConfig::new(action_vocab_size, 64, 128),
            ModelSize::Base => RnnModelConfig::new(action_vocab_size, 128, 256),
            ModelSize::Large => RnnModelConfig::new(action_vocab_size, 256, 512),
        }
    }
}

#[derive(Debug)]
pub struct RnnModel {
    action_embedding: nn::Embedding,
    rnn: nn::LSTM,
    output_proj: nn::Linear,
}

impl RnnModel {
    // Outputs unmasked logits.
    pub fn forward(
        &self,
        input: Tensor,
        hidden: &Option<LSTMState>, // [batch_size, seq_length, 10]
    ) -> (Tensor, Option<LSTMState>) {
        // FIXME : It implements Petersen's DSR(2021) agent, but we are adding semantic embeedding SNIP(2023).
        // states are Float. shape: [batch, 10] or [batch, seq, 10]
        let states_3d = if input.dim() == 2 {
            input.unsqueeze(1)
        } else {
            input.shallow_clone()
        };

        // RNN step
        let lstmstate = match hidden {
            Some(x) => x,
            None => &self.rnn.zero_state(states_3d.size()[0]),
        };
        let output = self.rnn.step(&states_3d, &lstmstate);
        let (rnn_out, _hidden_out) = (output.c(), output.h());

        // Output projection
        let logits = rnn_out.apply(&self.output_proj);
        (logits, Some(output))

        // // Mask invalid actions
        // available_actions: &Tensor, // [batch_size, seq_length, action_vocab_size], bool
        // let is_invalid = available_actions.logical_not();
        // logits.masked_fill(&is_invalid, f64::NEG_INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionSpace;
    use parser::behavior::test_make_behavior;
    use tch::{Device, Kind};

    #[test]
    fn test_rnn_inference_with_masking() {
        let behavior = test_make_behavior();
        let action_space: ActionSpace = (&behavior).into();
        let action_vocab_size = action_space.size();

        // 1. Initialize the RNN model
        let vs = nn::VarStore::new(Device::Cpu);
        let config = ModelSize::Small.get_config(action_vocab_size);
        let model = config.init(&vs.root());

        // 2. Create dummy state sequence [batch_size=1, seq_length=1, state_size=2]
        let states = Tensor::from_slice(&[1_i64, 2_i64]).view([1, 1, 2]);

        let mut mask = vec![false; action_vocab_size];
        for i in (0..action_vocab_size).step_by(2) {
            mask[i] = true;
        }
        let available_actions = Tensor::from_slice(&mask).view([1, 1, action_vocab_size as i64]);

        // 3. Run Forward Pass
        let lstm_state: Option<LSTMState> = None;
        let (logits_not_masked, _lstm_state) = model.forward(states, &lstm_state); // not masked
        let logits =
            logits_not_masked.masked_fill(&available_actions.logical_not(), f64::NEG_INFINITY);
        // available_actions: &Tensor, // [batch_size, seq_length, action_vocab_size], bool
        // let is_invalid = available_actions.logical_not();
        // logits.masked_fill(&is_invalid, f64::NEG_INFINITY)

        assert_eq!(logits.size(), vec![1, 1, action_vocab_size as i64]);

        // 6. Compute probabilities
        let probabilities = logits.softmax(-1, Kind::Float);

        // 7. Verify boundaries
        let probs_array: Vec<f32> = probabilities.squeeze_dims(&[0, 1]).try_into().unwrap();
        for i in (0..action_vocab_size).step_by(2) {
            assert!(probs_array[i] > 0.0);
            assert_eq!(probs_array[i + 1], 0.0);
        }

        let valid_sum: f32 = probs_array.iter().sum();
        assert!((valid_sum - 1.0).abs() < 1e-5);
    }
}
