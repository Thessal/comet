use burn::{
    backend::NdArray,
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::loss::CrossEntropyLossConfig,
    tensor::{
        Int, Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
    train::{ClassificationOutput, InferenceStep, LearningResult, TrainOutput, TrainStep},
};

use crate::model::TransformerModel;
use crate::search::{Action, EvaluatedSample, SearchEnv, SearchState};
use parser::program::{BehaviorDecl, Ident, TYPE_DECL_LENGTH, TypeDecl};

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
    behavior: &parser::program::BehaviorDecl,
    available_funcs: &[(Ident, Vec<TypeDecl>, TypeDecl)],
) -> usize {
    let ints = behavior
        .integers
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let floats = behavior
        .floats
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let strings = behavior
        .strings
        .as_ref()
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    let base_ints = 2; // Done is 0, Shift is 1
    let base_floats = base_ints + ints.len();
    let base_strings = base_floats + floats.len();
    let base_funcs = base_strings + strings.len();

    match action {
        Action::Done => 0,
        Action::Shift => 1,
        Action::ShiftInteger(v) => {
            let idx = ints.iter().position(|x| x == v).unwrap_or(0);
            base_ints + idx
        }
        Action::ShiftFloat(v) => {
            let idx = floats
                .iter()
                .position(|x| (x - v).abs() < 1e-6)
                .unwrap_or(0);
            base_floats + idx
        }
        Action::ShiftString(v) => {
            let idx = strings.iter().position(|x| x == v).unwrap_or(0);
            base_strings + idx
        }
        Action::Reduce(func_name) => {
            let idx = available_funcs
                .iter()
                .position(|(n, _, _)| n == func_name)
                .expect(&format!(
                    "Function {} not found in available_funcs",
                    func_name
                ));
            base_funcs + idx
        }
    }
}

pub fn string_to_action(s: &str) -> Action {
    if s == "!shift" {
        Action::Shift
    } else if s == "!done" {
        Action::Done
    } else if s.starts_with("\"") {
        Action::ShiftString(s.trim_matches('"').to_string())
    } else if let Ok(i) = s.parse::<i64>() {
        Action::ShiftInteger(i)
    } else if let Ok(f) = s.parse::<f64>() {
        Action::ShiftFloat(f)
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

        let env = SearchEnv::new(
            target_type,
            behavior.integers.clone().unwrap_or_default(),
            behavior.floats.clone().unwrap_or_default(),
            behavior.strings.clone().unwrap_or_default(),
            false,
        );

        for sample in samples {
            // Replay the sample sequence to extract state action pairs
            let mut current_state = initial_state.clone();
            let mut states = Vec::new();
            let mut targets = Vec::new();

            let mut prev_action_id = 0; // Pad/Unknown for first step

            for a in &sample.actions {
                let action = string_to_action(a);
                let target_id = action_to_id(&action, behavior, available_funcs);

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
            let target_id = action_to_id(&Action::Done, behavior, available_funcs);
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

pub fn train(
    behavior: &parser::program::BehaviorDecl,
    available_funcs: &[(
        parser::program::Ident,
        Vec<parser::program::TypeDecl>,
        parser::program::TypeDecl,
    )],
    samples: &[crate::search::EvaluatedSample],
    num_epochs: usize,
    batch_size: usize,
    num_workers: usize,
) -> crate::model::TransformerModel<burn::backend::ndarray::NdArray> {
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

    type BackendBase = NdArray<f32>; // TODO: move this to model configuration (model.rs or config.rs)
    type BackendAutoDiff = Autodiff<BackendBase>;
    let device = <BackendBase as Backend>::Device::default();

    let batcher_train = TransformerBatcher::new(32);
    let batcher_valid = TransformerBatcher::new(32);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset.clone());

    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(dataset);

    println!("--- Initializing Transformer Model ---");
    let type_vocab_size = 1 + TYPE_DECL_LENGTH; // 0 = PAD
    let action_vocab_size = 2 // Done and Shift
        + behavior.floats.as_ref().map_or(0, |v| v.len())
        + behavior.integers.as_ref().map_or(0, |v| v.len())
        + behavior.strings.as_ref().map_or(0, |v| v.len())
        + available_funcs.len();
    let config = ModelSize::Small.get_config(type_vocab_size, action_vocab_size);
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
    trained.model
}

pub fn generate(
    behavior: &parser::program::BehaviorDecl,
    available_funcs: &[(
        parser::program::Ident,
        Vec<parser::program::TypeDecl>,
        parser::program::TypeDecl,
    )],
    inference_model: &crate::model::TransformerModel<burn::backend::ndarray::NdArray>,
    temperature: f64, // 1.0 is base, <1.0 is more deterministic, >1.0 is more random
) -> Vec<String> {
    type BackendBase = NdArray<f32>; // TODO: move this to model configuration (model.rs or config.rs
    let device = <BackendBase as burn::tensor::backend::Backend>::Device::default();

    println!("--- Running Inference Test ---");

    use burn::tensor::Tensor;

    let env = crate::search::SearchEnv::new(
        behavior.return_type.clone(),
        behavior.integers.clone().unwrap_or_default(),
        behavior.floats.clone().unwrap_or_default(),
        behavior.strings.clone().unwrap_or_default(),
        true, // insert constants only when unprocessed param length < 3
    );
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
            .map(|action_str| {
                action_to_id(&string_to_action(action_str), behavior, available_funcs)
            })
            .unwrap_or(0);

        let encoded = encode_state(&state, prev_action_id);
        let encoded_i32: Vec<i32> = encoded.iter().map(|&x| x as i32).collect();

        let input_tensor = Tensor::<BackendBase, 3, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(encoded_i32, [1, 1, 8]),
            &device,
        );

        let logits = inference_model.forward(input_tensor);
        println!("Logits: {:?}", logits);
        println!("Logits / temperature: {:?}", logits.clone() / temperature);
        let probs = burn::tensor::activation::softmax(logits / temperature, 2);
        println!("Probs: {:?}", probs);

        // Extract softmax probabilities. For NdArray backend, into_data().to_vec::<f32>() is safe.
        let probs_data = probs.into_data().to_vec::<f32>().unwrap();

        let valid_candidates = valid_actions;
        if valid_candidates.is_empty() {
            break;
        }

        // Action selection: Using the Transformer's learned probabilities -> Weighted Sampling
        let mut valid_probs: Vec<f32> = valid_candidates
            .iter()
            .map(|a| {
                let id = action_to_id(a, behavior, available_funcs);
                probs_data.get(id).copied().unwrap_or(0.0)
            })
            .collect();

        // Rescale sum of valid probabilities to 1
        let sum: f32 = valid_probs.iter().sum();
        if sum > 1e-6 {
            for p in &mut valid_probs {
                *p /= sum;
            }
        } else {
            let uniform = 1.0 / valid_candidates.len() as f32;
            for p in &mut valid_probs {
                *p = uniform;
            }
        }
        println!("valid_candidates: {:?}", valid_candidates);
        println!("valid_probs: {:?}", valid_probs);

        use rand::distributions::WeightedIndex;
        use rand::prelude::*;
        let dist = WeightedIndex::new(&valid_probs).unwrap();
        let mut rng = thread_rng();
        let chosen_action = valid_candidates[dist.sample(&mut rng)].clone();

        if chosen_action == crate::search::Action::Done {
            break;
        }

        state = env.step(&state, chosen_action, available_funcs).unwrap();

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
            operators: None,
            integers: None,
            floats: None,
            strings: None,
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
            operators: None,
            integers: None,
            floats: None,
            strings: None,
        };
        let sample = EvaluatedSample {
            actions: vec![
                "!shift".to_string(),
                "!shift".to_string(),
                "add".to_string(),
            ],
            fitness: vec![1.0],
        };
        let trained = train(
            &behavior,
            &available_funcs,
            &[sample.clone(), sample],
            1, //epoch
            2, //bs
            1, //worker
        );
        let _generated = generate(&behavior, &available_funcs, &trained, 1.0);
    }
}
