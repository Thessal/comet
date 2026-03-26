use burn::{
    data::dataloader::batcher::Batcher,
    data::dataset::Dataset,
    nn::loss::CrossEntropyLossConfig,
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend, backend::Backend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::model::TransformerModel;
use crate::search::{Action, EvaluatedSample, SearchEnv, SearchState};
use parser::program::{BehaviorDecl, Ident, TypeDecl};

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

//----------------------------------------------------------------------
// Vocabulary and Encoding
//----------------------------------------------------------------------

pub fn type_to_id(t: &TypeDecl) -> usize {
    match t {
        TypeDecl::DataFrame => 1,
        TypeDecl::Matrix => 2,
        TypeDecl::Vector => 3,
        TypeDecl::String => 4,
        TypeDecl::Float => 5,
        TypeDecl::Bool => 6,
        TypeDecl::Void => 7,
    }
}

pub fn action_to_id(
    action: &Action,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
) -> usize {
    match action {
        Action::Done => 0,
        Action::Shift => 1,
        Action::Reduce(func_name) => {
            let idx = available_funcs
                .iter()
                .position(|(n, _, _)| n == func_name)
                .expect(&format!(
                    "Function {} not found in available_funcs",
                    func_name
                ));
            2 + idx
        }
    }
}

pub fn string_to_action(s: &str) -> Action {
    if s == "!shift" {
        Action::Shift
    } else if s == "!done" {
        Action::Done
    } else {
        Action::Reduce(s.to_string())
    }
}

pub fn encode_state(state: &SearchState, prev_action_id: usize) -> [usize; 8] {
    let mut encoded = [0; 8];
    encoded[0] = prev_action_id;

    // next unprocessed parameter
    if let Some(param) = state.unprocessed_params.last() {
        encoded[1] = type_to_id(param);
    }

    // up to 6 stack items
    let stack_len = state.stack.len();
    for i in 0..6 {
        if i < stack_len {
            encoded[2 + i] = type_to_id(&state.stack[stack_len - 1 - i]);
        }
    }

    encoded
}

//----------------------------------------------------------------------
// Dataset and Batching
//----------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TransformerBatchItem {
    pub states: Vec<[usize; 8]>,
    pub targets: Vec<usize>,
}

#[derive(Clone)]
pub struct ProgramSequenceDataset {
    items: Vec<TransformerBatchItem>,
}

impl ProgramSequenceDataset {
    pub fn new(
        samples: &[EvaluatedSample],
        behavior: &BehaviorDecl,
        available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    ) -> Self {
        let mut items = Vec::new();

        let target_type = behavior.return_type.clone();
        let unprocessed_params: Vec<TypeDecl> = behavior
            .args
            .iter()
            .rev()
            .map(|arg| arg.type_decl.clone())
            .collect();

        let initial_state = SearchState {
            unprocessed_params,
            stack: vec![],
            sequence: vec![],
        };

        let env = SearchEnv::new(target_type);

        for sample in samples {
            // Replay the sample sequence to extract state action pairs
            let mut current_state = initial_state.clone();
            let mut states = Vec::new();
            let mut targets = Vec::new();

            let mut prev_action_id = 0; // Pad/Unknown for first step

            for action_str in &sample.actions {
                let action = string_to_action(action_str);
                let target_id = action_to_id(&action, available_funcs);

                let state_tensor = encode_state(&current_state, prev_action_id);
                states.push(state_tensor);
                targets.push(target_id);

                // Advance state
                current_state = env
                    .step(&current_state, action, available_funcs)
                    .expect("Failed to step environment during dataset creation");

                prev_action_id = target_id;
            }

            // Finally, add the Done action target!
            let target_id = action_to_id(&Action::Done, available_funcs);
            let state_tensor = encode_state(&current_state, prev_action_id);
            states.push(state_tensor);
            targets.push(target_id);

            items.push(TransformerBatchItem { states, targets });
        }

        Self { items }
    }
}

impl Dataset<TransformerBatchItem> for ProgramSequenceDataset {
    fn get(&self, index: usize) -> Option<TransformerBatchItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct TransformerBatcher {
    max_seq_len: usize,
}

impl TransformerBatcher {
    pub fn new(max_seq_len: usize) -> Self {
        Self { max_seq_len }
    }
}

impl<B: Backend> Batcher<B, TransformerBatchItem, TransformerBatch<B>> for TransformerBatcher {
    fn batch(&self, items: Vec<TransformerBatchItem>, device: &B::Device) -> TransformerBatch<B> {
        let batch_size = items.len();

        let mut inputs_data = Vec::with_capacity(batch_size * self.max_seq_len * 8);
        let mut targets_data = Vec::with_capacity(batch_size * self.max_seq_len);

        for item in items.iter() {
            let seq_len = item.states.len();
            for t in 0..self.max_seq_len {
                if t < seq_len {
                    inputs_data.extend_from_slice(&item.states[t]);
                    targets_data.push(item.targets[t] as i32);
                } else {
                    // Padding: states = [0; 8], target = 0 (Done, or some ignore index)
                    inputs_data.extend_from_slice(&[0; 8]);
                    targets_data.push(0);
                }
            }
        }

        let inputs_tensor = Tensor::<B, 3, Int>::from_data(
            TensorData::new(
                inputs_data
                    .into_iter()
                    .map(|v| v as i32)
                    .collect::<Vec<_>>(),
                [batch_size, self.max_seq_len, 8],
            ),
            device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets_data, [batch_size, self.max_seq_len]),
            device,
        );

        TransformerBatch::new(inputs_tensor, targets_tensor)
    }
}

pub fn train_and_sample(
    behavior: &BehaviorDecl,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
    samples: &[EvaluatedSample],
    num_epochs: usize,
) -> Vec<String> {
    use crate::model::ModelSize;
    use burn::backend::Autodiff;
    use burn::backend::ndarray::NdArray;
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::optim::{AdamConfig, Optimizer};
    use burn::tensor::backend::Backend;
    use burn::train::metric::LossMetric;
    use burn::train::{Learner, SupervisedTraining};

    println!("--- Building Program Sequence Dataset ---");
    let dataset = ProgramSequenceDataset::new(samples, behavior, available_funcs);

    type BackendBase = NdArray<f32>;
    type BackendAutoDiff = Autodiff<BackendBase>;
    let device = <BackendBase as Backend>::Device::default();

    let batcher_train = TransformerBatcher::new(32);
    let batcher_valid = TransformerBatcher::new(32);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(2)
        .num_workers(1)
        .build(dataset.clone());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(2)
        .num_workers(1)
        .build(dataset);

    println!("--- Initializing Transformer Model ---");
    let config = ModelSize::Small.get_config(15, 20);
    let model = config.init::<BackendAutoDiff>(&device);
    let config_optim = AdamConfig::new();

    let artifact_dir = "/tmp/comet-supervised-rl-test";
    std::fs::remove_dir_all(artifact_dir).ok();

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(burn::record::CompactRecorder::new())
        .num_epochs(num_epochs)
        .summary();

    println!("--- Launching Target Supervised Training ---");
    let trained = training.launch(Learner::new(model, config_optim.init(), 1e-4));
    println!("--- Training Completed ---");

    println!("--- Running Inference Test ---");
    let inference_model = trained.model;

    use burn::tensor::Tensor;

    let env = crate::search::SearchEnv::new(behavior.return_type.clone());
    let mut state = crate::search::SearchState {
        unprocessed_params: behavior
            .args
            .iter()
            .rev()
            .map(|arg| arg.type_decl.clone())
            .collect(),
        stack: vec![],
        sequence: vec![],
    };

    for step_count in 0..100 {
        let valid_actions = env.get_valid_actions(&state, available_funcs);
        if valid_actions.is_empty() {
            break;
        }

        if valid_actions.contains(&crate::search::Action::Done) {
            break;
        }

        let prev_action_id = state
            .sequence
            .last()
            .map(|action_str| action_to_id(&string_to_action(action_str), available_funcs))
            .unwrap_or(0);

        let encoded = encode_state(&state, prev_action_id);
        let encoded_i32: Vec<i32> = encoded.iter().map(|&x| x as i32).collect();

        let input_tensor = Tensor::<BackendBase, 3, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(encoded_i32, [1, 1, 8]),
            &device,
        );

        let logits = inference_model.forward(input_tensor);
        let probs = burn::tensor::activation::softmax(logits, 2);
        
        // Extract softmax probabilities. For NdArray backend, into_data().to_vec::<f32>() is safe.
        let probs_data = probs.into_data().to_vec::<f32>().unwrap();

        let valid_non_done: Vec<_> = valid_actions
            .into_iter()
            .filter(|a| *a != crate::search::Action::Done)
            .collect();
            
        if valid_non_done.is_empty() {
            break;
        }

        // Action selection: Using the Transformer's learned probabilities -> Argmax over Valid Actions
        let best_action = valid_non_done
            .into_iter()
            .max_by(|a, b| {
                let id_a = action_to_id(a, available_funcs);
                let id_b = action_to_id(b, available_funcs);
                
                // Ensure we don't out-of-bounds index since vocab config might differ.
                let p_a = probs_data.get(id_a).copied().unwrap_or(0.0);
                let p_b = probs_data.get(id_b).copied().unwrap_or(0.0);
                
                p_a.partial_cmp(&p_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        state = env.step(&state, best_action, available_funcs).unwrap();
        
        if step_count == 99 {
            println!("Warning: Inference loop hit 100 max iterations.");
        }
    }

    state.sequence
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use parser::program::TypedArg;

    type B = NdArray<f32>;

    #[test]
    fn test_encode_and_batch() {
        let available_funcs = vec![(
            "add".to_string(),
            vec![TypeDecl::Float, TypeDecl::Float],
            TypeDecl::Float,
        )];

        let behavior = BehaviorDecl {
            name: "TestBehavior".to_string(),
            args: vec![
                TypedArg {
                    name: "a".to_string(),
                    type_decl: TypeDecl::Float,
                },
                TypedArg {
                    name: "b".to_string(),
                    type_decl: TypeDecl::Float,
                },
            ],
            return_type: TypeDecl::Float,
            weights: None,
            train: None,
            supervised_samples: None,
        };

        // We simulate a valid generation
        // Unprocessed at start: [Float(b), Float(a)]
        let sample = EvaluatedSample {
            actions: vec![
                "!shift".to_string(),
                "!shift".to_string(),
                "add".to_string(),
            ],
            fitness: vec![1.0],
        };

        let dataset = ProgramSequenceDataset::new(&[sample], &behavior, &available_funcs);
        assert_eq!(dataset.len(), 1);

        let item = dataset.get(0).unwrap();
        // 3 actions + 1 Done action = 4 sequence steps
        assert_eq!(item.states.len(), 4);
        assert_eq!(item.targets.len(), 4);

        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let batcher = TransformerBatcher::new(8); // pad to 8

        let batch: TransformerBatch<B> = batcher.batch(vec![item], &device);

        assert_eq!(batch.inputs.dims(), [1, 8, 8]);
        assert_eq!(batch.targets.dims(), [1, 8]);
    }

    #[test]
    fn test_burn_training_and_inference_example() {
        let available_funcs = vec![(
            "add".to_string(),
            vec![TypeDecl::Float, TypeDecl::Float],
            TypeDecl::Float,
        )];
        let behavior = BehaviorDecl {
            name: "TestBehavior".to_string(),
            args: vec![
                parser::program::TypedArg {
                    name: "a".to_string(),
                    type_decl: TypeDecl::Float,
                },
                parser::program::TypedArg {
                    name: "b".to_string(),
                    type_decl: TypeDecl::Float,
                },
            ],
            return_type: TypeDecl::Float,
            weights: None,
            train: None,
            supervised_samples: None,
        };
        let sample = EvaluatedSample {
            actions: vec![
                "!shift".to_string(),
                "!shift".to_string(),
                "add".to_string(),
            ],
            fitness: vec![1.0],
        };
        let _generated =
            train_and_sample(&behavior, &available_funcs, &[sample.clone(), sample], 1);
    }
}
