use parser::parser::Rule::list_identifier;
use runtime::runtime::Runtime;
use stdlib::types::Signal;
use tch::{
    Device,
    Kind::Float,
    Tensor,
    nn::{self, LSTMState, Module, RNN},
};

use crate::{action::Action, action::ActionSpace, state::SearchState};

// pub struct Model {}
// pub enum ModelConfig {
//     RnnModel(RnnModelConfig),
//     // TransformerModel(TransformerModelConfig),
// }

// #[derive(Debug)]
// pub enum Model {
//     RnnModel(RnnModel),
//     // TransformerModel(TransformerModel)
// }

pub trait Model {
    fn reset(&self) {
        // resets internal state of time series models
        unimplemented!()
    }
    fn forward(
        &mut self,
        state: &SearchState,
        masks: &Tensor,
        device: &Device,
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
        state: &SearchState,
        masks: &Tensor,
        device: &Device,
    ) -> (Tensor, Tensor) {
        // (state_embedding, action_logits)}
        let logits = tch::Tensor::ones(
            [1, self.action_space.size() as i64],
            (tch::Kind::Float, *device),
        );
        let dummy_emb = tch::Tensor::zeros([1, 1], (tch::Kind::Float, *device));
        let masked_logits = logits.masked_fill(&masks.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);
        (dummy_emb, masked_logits)
    }
}

pub struct LstmModel {
    data_embedding_model: nn::Embedding,
    policy_model: nn::LSTM,
    lstm_hidden: LSTMState,
    output_proj: nn::Linear,
    runtime: Runtime,
}

impl Model for LstmModel {
    fn reset(&self) {
        todo!() // Initilaize lstm_hidden
    }
    fn forward(
        &mut self,
        state: &SearchState,
        masks: &Tensor,
        device: &Device,
    ) -> (Tensor, Tensor) {
        // (state_embedding, action_logits)

        // assert_eq!(input.dim(), 2); // [batch_size, 10]
        // assert_eq!(input.size()[1], crate::env::EMBEDDING_SIZE as i64);

        // get data
        let (stack, callgraph) = state.machine.get_stack();
        let mut data: Vec<Tensor> = Vec::new();
        let runtime = &mut self.runtime;
        for (_signal_decl, addr) in stack.iter() {
            let signal = runtime.lookup_or_run(callgraph, *addr);
            match signal {
                Signal::DataFrame(Some(df)) => data.push(df.shallow_clone()),
                _ => {
                    continue;
                }
            }
        }

        // stack data embedding
        let stack_embeddings: Vec<Tensor> = data
            .iter()
            .map(|df| self.data_embedding_model.forward(df))
            .collect();
        let stack_embedding =
            Tensor::stack(&stack_embeddings, 0).sum_dim_intlist(&vec![0], false, Float);

        // symbol embedding
        let stack_types: Vec<f64> = stack
            .iter()
            .map(|(signal_decl, _addr)| Into::<usize>::into(signal_decl) as f64)
            .collect();
        let stack_types: Tensor = Tensor::from_slice(&stack_types).to_kind(Float).to(*device);

        // action
        let state_embedding = Tensor::concat(&[stack_embedding, stack_types], 0)
            .to_kind(Float)
            .to(*device);
        self.lstm_hidden = self.policy_model.step(&state_embedding, &self.lstm_hidden);
        let policy: Tensor = self.lstm_hidden.h();
        let logits = self.output_proj.forward(&policy); // Output projection

        // apply mask
        todo!();
        (state_embedding, logits)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::action::ActionSpace;
//     use parser::behavior::test_make_behavior;
//     use tch::{Device, Kind};

//     #[test]
//     fn test_rnn_inference_with_masking() {
//         let behavior = test_make_behavior();
//         let action_space: ActionSpace = (&behavior).into();
//         let action_vocab_size = action_space.size();

//         // 1. Initialize the RNN model
//         let vs = nn::VarStore::new(Device::Cpu);
//         let config = ModelSize::Small.get_config(action_vocab_size);
//         let model = config.init(&vs.root());

//         // 2. Create dummy state sequence [batch_size=1, state_size=crate::env::EMBEDDING_SIZE]
//         let states = Tensor::from_slice(&[1_i64; crate::env::EMBEDDING_SIZE])
//             .view([1, crate::env::EMBEDDING_SIZE as i64]);

//         let mut mask = vec![false; action_vocab_size];
//         for i in (0..action_vocab_size).step_by(2) {
//             mask[i] = true;
//         }
//         let available_actions = Tensor::from_slice(&mask).view([1, action_vocab_size as i64]);

//         // 3. Run Forward Pass
//         let lstm_state: Option<LSTMState> = None;
//         let (logits_not_masked, _lstm_state) = model.forward(states, &lstm_state); // not masked
//         assert_eq!(logits_not_masked.size()[1], action_vocab_size as i64);
//         let logits =
//             logits_not_masked.masked_fill(&available_actions.logical_not(), f64::NEG_INFINITY);

//         assert_eq!(logits.size(), vec![1, action_vocab_size as i64]);

//         // 6. Compute probabilities
//         let probabilities = logits.softmax(-1, Kind::Float);

//         // 7. Verify boundaries
//         let probs_array: Vec<f32> = probabilities.squeeze_dim(0).try_into().unwrap();
//         for i in (0..action_vocab_size).step_by(2) {
//             assert!(probs_array[i] > 0.0);
//             assert_eq!(probs_array[i + 1], 0.0);
//         }

//         let valid_sum: f32 = probs_array.iter().sum();
//         assert!((valid_sum - 1.0).abs() < 1e-5);
//     }
// }
