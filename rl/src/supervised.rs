use burn::{
    data::dataset::Dataset,
    nn::loss::CrossEntropyLossConfig,
    tensor::{Int, Tensor, backend::AutodiffBackend, backend::Backend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::model::TransformerModel;

/// The input batch provided to the model during training.
#[derive(Clone, Debug)]
pub struct TransformerBatch<B: Backend> {
    pub inputs: Tensor<B, 3, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> TransformerBatch<B> {
    pub fn new(inputs: Tensor<B, 3, Int>, targets: Tensor<B, 2, Int>) -> Self {
        Self { inputs, targets }
    }
}

impl<B: AutodiffBackend> TrainStep for TransformerModel<B> {
    type Input = TransformerBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TransformerBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward(batch.inputs);
        let [batch_size, seq_length, vocab_size] = output.dims();
        let device = output.device();

        // Flatten sequence dimension
        let output_flat = output.reshape([batch_size * seq_length, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output_flat.clone(), targets_flat.clone());

        TrainOutput::new(
            self,
            loss.backward(),
            ClassificationOutput {
                loss,
                output: output_flat,
                targets: targets_flat,
            },
        )
    }
}

impl<B: Backend> InferenceStep for TransformerModel<B> {
    type Input = TransformerBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: TransformerBatch<B>) -> ClassificationOutput<B> {
        let output = self.forward(batch.inputs);
        let [batch_size, seq_length, vocab_size] = output.dims();
        let device = output.device();

        let output_flat = output.reshape([batch_size * seq_length, vocab_size]);
        let targets_flat = batch.targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(output_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: output_flat,
            targets: targets_flat,
        }
    }
}

pub struct AstSequenceDataset {
    sequences: Vec<Vec<usize>>,
}

impl Dataset<Vec<usize>> for AstSequenceDataset {
    fn get(&self, index: usize) -> Option<Vec<usize>> {
        self.sequences.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.sequences.len()
    }
}

// ... Batcher implementation omitted for brevity, will implement if required ...
