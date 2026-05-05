use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, Lstm, LstmConfig},
    tensor::{Tensor, backend::Backend},
};

pub enum ModelConfig {
    RnnModel(RnnModelConfig),
    // TransformerModel(TransformerModelConfig),
}

#[derive(Module, Debug)]
pub enum Model<B: Backend> {
    RnnModel(RnnModel<B>),
    // TransformerModel(TransformerModel<B>)
}

impl<B: Backend> Model<B> {
    pub fn forward(
        &self,
        states: Tensor<B, 3, burn::tensor::Int>,
        available_actions: Tensor<B, 3, burn::tensor::Bool>,
    ) -> Tensor<B, 3> {
        match self {
            Model::RnnModel(model) => model.forward(states, available_actions),
        }
    }
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        match self {
            ModelConfig::RnnModel(config) => Model::RnnModel(config.init(device)),
        }
    }
}

#[derive(Config, Debug)]
pub struct RnnModelConfig {
    pub action_vocab_size: usize,
    pub d_model: usize,
    pub d_hidden: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl RnnModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RnnModel<B> {
        let action_embedding =
            EmbeddingConfig::new(self.action_vocab_size, self.d_model).init(device);

        // RNN takes the concatenation of parent and sibling embeddings
        let rnn = LstmConfig::new(self.d_model * 2, self.d_hidden, true).init(device);

        let output_proj = LinearConfig::new(self.d_hidden, self.action_vocab_size).init(device);

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

#[derive(Module, Debug)]
pub struct RnnModel<B: Backend> {
    action_embedding: Embedding<B>,
    rnn: Lstm<B>,
    output_proj: Linear<B>,
}

impl<B: Backend> RnnModel<B> {
    pub fn forward(
        &self,
        states: Tensor<B, 3, burn::tensor::Int>,
        available_actions: Tensor<B, 3, burn::tensor::Bool>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, _] = states.dims();

        // 1. Slice parent and sibling
        // Assuming states[.., .., 0] is Parent and states[.., .., 1] is Sibling
        let parent = states
            .clone()
            .slice([0..batch_size, 0..seq_length, 0..1])
            .reshape([batch_size, seq_length]);

        let sibling = states
            .slice([0..batch_size, 0..seq_length, 1..2])
            .reshape([batch_size, seq_length]);

        // 2. Embed
        let parent_emb = self.action_embedding.forward(parent);
        let sibling_emb = self.action_embedding.forward(sibling);

        // 3. RNN step
        let rnn_input = Tensor::cat(vec![parent_emb, sibling_emb], 2);
        let (rnn_out, _) = self.rnn.forward(rnn_input, None);

        // 4. Output projection to Library (action space)
        let logits = self.output_proj.forward(rnn_out);

        // 5. Mask out invalid actions
        let is_invalid = available_actions.bool_not();
        assert_eq!(is_invalid.shape(), logits.shape());
        logits.mask_fill(is_invalid, -f32::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use crate::action::ActionSpace;

    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::{Tensor, TensorData};
    use parser::behavior::test_make_behavior;

    type Backend = NdArray<f32>;

    #[test]
    fn test_rnn_inference_with_masking() {
        // TODO: review this machine-generated code
        let device = Default::default();
        let behavior = test_make_behavior();
        let action_space: ActionSpace = (&behavior).into();
        let action_vocab_size = action_space.size();

        // 1. Initialize the RNN model
        let config = ModelSize::Small.get_config(action_vocab_size);
        let model = config.init::<Backend>(&device);

        // 2. Create a dummy state sequence (batch_size=1, seq_length=1, state_size=2)
        // Format: [Parent, Sibling]
        let state_data = TensorData::from([[
            [1, 2], // parent=1, sibling=2
        ]]);
        let states = Tensor::<Backend, 3, burn::tensor::Int>::from_data(
            state_data.convert::<i32>(),
            &device,
        );

        let mut mask = vec![false].repeat(action_vocab_size);
        for i in (0..action_vocab_size).step_by(2) {
            mask[i] = true;
        }
        let available_mask = Tensor::<Backend, 3, burn::tensor::Bool>::from_bool(
            burn::tensor::TensorData::new(mask, [1, 1, action_vocab_size]),
            &device,
        );
        let available_actions = available_mask;

        // 3. Run Forward Pass
        let logits = model.forward(states, available_actions);

        // Output logits shape should be [1, 1, action_vocab_size]
        assert_eq!(logits.dims(), [1, 1, action_vocab_size]);

        // 6. Compute action probabilities
        let probabilities = burn::tensor::activation::softmax(logits, 2);

        // 7. Verify math boundaries
        let tensor_data = probabilities.into_data();
        let probs_array: &[f32] = tensor_data.as_slice().unwrap();
        for i in (0..action_vocab_size).step_by(2) {
            assert!(probs_array[i] > 0.0);
            assert_eq!(probs_array[i + 1], 0.0);
        }

        let valid_sum: f32 = probs_array.iter().sum();
        assert!((valid_sum - 1.0).abs() < 1e-5);
    }
}
