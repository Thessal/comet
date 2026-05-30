// TODO: dataframe embedding model
// I guess log transform and CNN will work?
use tch::Tensor;

use crate::state::SearchState;

static EMBEDDING_SIZE_PER_TOKEN: usize = 5; // 5 floats per token
static EMBEDDING_TOKEN_CNT: usize = 2; // two tokens
pub static EMBEDDING_SIZE: usize = EMBEDDING_SIZE_PER_TOKEN * EMBEDDING_TOKEN_CNT;
// pub static EMBEDDER: Option<EmbeddingModel> = None;

struct EmbeddingModel {
    embedding: tch::nn::Embedding,
}

impl EmbeddingModel {
    pub fn new() -> Self {
        todo!()
    }
    pub fn state_embed(&self, state: &SearchState, device: tch::Device) -> tch::Tensor {
        // Petersen(2021): last two token, 1 float per token.
        let mut embedding_tokens: Vec<Tensor> = vec![
            Tensor::from_slice(&[0.0f64; EMBEDDING_SIZE_PER_TOKEN]),
            Tensor::from_slice(&[0.0f64; EMBEDDING_SIZE_PER_TOKEN]),
        ];
        // for (i, tok) in state.expr.iter().take(EMBEDDING_TOKEN_CNT).enumerate() {
        //     embedding_tokens[i] = runtime::ast::token_to_tensor(tok); // FIXME
        // }
        // assert!(embedding_tokens.len() == EMBEDDING_TOKEN_CNT);
        // let out = tch::Tensor::cat(&embedding_tokens, 0).to_device(device);
        // assert!(out.dim() == 1);
        // assert!(out.size()[0] as usize == EMBEDDING_SIZE);

        // TODO
        // for now simply output zero
        let out: Tensor = tch::Tensor::zeros(&[EMBEDDING_SIZE as i64], (tch::Kind::Float, device));
        out
    }
}

// // TODO:
// // SNIP (2023) paper used tokenization and attention pooling.
// // This is simplified, max pooling based embedding. Let's try this first.
// let data_size = self.runtime.dmgr.data_size;
// let embeddings: Vec<Vec<Vec<f64>>> = state
//     .stack
//     .iter()
//     .map(|(_, _, signal)| signal.to_dataframe(data_size))
//     .collect();
// todo!("data_embedding_model need to be implemented");
// let embeddings: Vec<tch::Tensor> = embeddings
//     .into_iter()
//     .map(|x| data_embedding_model(&x).to_device(device))
//     .collect();
// tch::Tensor::stack(&embeddings, 1).max_dim(1, false).0 // max pooling
//     }
// }
