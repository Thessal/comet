use clap::Parser;
use parser::behavior::{BehaviorDecl, Declaration};
use parser::parser::parse;
use std::fs;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input .cm file to compile
    file: String,

    /// Optional flag to use CUDA backend
    #[arg(long)]
    cuda: bool,
}

fn generate_valid_expressions(
    behavior: &BehaviorDecl,
    available_funcs: &Vec<runtime::ast::OperatorSpec>,
    k: usize,
    runtime: &mut runtime::runtime::Runtime,
) -> Vec<rl::action::EvaluatedSample> {
    // TODO: generate top k and drop duplicate samples. the result count may be smaller than k
    println!("Inferring action candidates from the code:");
    for f in available_funcs {
        println!("- {}({:?}) -> {:?}", f.name, f.inputs, f.output);
    }

    println!("\nGenerating sample expression trees and evaluating using runtime...");
    let samples = rl::action::generate_top_k_samples(
        &behavior,
        available_funcs,
        k,
        |fitness| fitness[0] > 0.,
        runtime,
    );

    for (i, sample) in samples.iter().enumerate() {
        println!("Sample {}: Fitness = {:?}", i + 1, sample.fitness);
        println!("  Actions: {:?}", sample.actions);
    }

    assert!(
        !samples.is_empty(),
        "Should generate at least one valid expression tree"
    );
    samples
}

fn main() {
    let args = Args::parse();
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let program = parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    // Select first train=True Behavior
    let mut target_behavior = None;
    for decl in &program.declarations {
        if let Declaration::Behavior(b) = decl {
            if Some(true) == b.train {
                target_behavior = Some(b.clone());
            }
        }
    }
    let behavior = target_behavior.expect("No train=True behavior found");
    println!("--- Selected behavior : {:?} ---", behavior.name);

    let available_funcs = if let Some(available_funcs_str) = &behavior.operators {
        let mut available_funcs = Vec::new();
        for func_str in available_funcs_str {
            available_funcs.push(runtime::ast::OperatorSpec::get_available_func(&func_str));
        }
        available_funcs
    } else {
        runtime::ast::OperatorSpec::get_available_funcs()
    };
    println!("--- Available functions : {:?} ---", available_funcs);

    // Initialize central runtime
    let mut runtime = runtime::runtime::Runtime::new(10000, "data");

    // Extract bound parameter values from the Flow call syntax
    let mut call_args = behavior
        .args
        .iter()
        .map(|_arg| "volume".to_string())
        .collect::<Vec<_>>();

    for decl in &program.declarations {
        if let Declaration::Flow(f) = decl {
            for stmt in &f.body {
                if let parser::behavior::FlowStmt::Expr(parser::behavior::Expr::Call {
                    path,
                    args,
                }) = stmt
                {
                    if path.segments.join("::") == behavior.name {
                        let mut p = Vec::new();
                        for arg in args {
                            match arg {
                                parser::behavior::Expr::Identifier(name) => p.push(name.clone()),
                                parser::behavior::Expr::Literal(
                                    parser::behavior::Literal::Float(f),
                                ) => p.push(f.to_string()),
                                parser::behavior::Expr::Literal(
                                    parser::behavior::Literal::Integer(i),
                                ) => p.push(i.to_string()),
                                _ => p.push("volume".to_string()),
                            }
                        }
                        call_args = p;
                    }
                }
            }
        }
    }

    // `evaluate_sequence` pops from the end, so we MUST reverse `call_args` to match the Shift order!
    call_args.reverse();

    if use_cuda {
        println!("--- Using CUDA backend ---");
        run_with_backend::<burn::backend::Autodiff<burn::backend::Cuda>>(
            &behavior,
            &available_funcs,
            runtime,
            call_args,
        );
    } else {
        println!("--- Using NdArray backend ---");
        run_with_backend::<burn::backend::Autodiff<burn::backend::ndarray::NdArray>>(
            &behavior,
            &available_funcs,
            runtime,
            call_args,
        );
    }
}

fn run_with_backend<B: burn::tensor::backend::AutodiffBackend>(
    behavior: &BehaviorDecl,
    available_funcs: &Vec<runtime::ast::OperatorSpec>,
    mut runtime: runtime::runtime::Runtime,
    call_args: Vec<String>,
) {
    // 1) Build dataset, 2) Train transformer,
    let sample = generate_valid_expressions(behavior, available_funcs, 100, &mut runtime);
    let trained_model = rl::supervised::train::<B>(
        behavior,
        available_funcs,
        &sample,
        behavior.supervised_epochs.unwrap(),
        32, //bs
        16, //num_worker
    );
    let _ = std::io::stdout().flush();
    println!(
        "Cache hit rate: {} / {}",
        runtime.expr_hits, runtime.expr_lookups
    );

    // -- RL Fine-Tuning Step --
    println!("--- Starting RL Fine-Tuning ---");
    use burn::module::{AutodiffModule, Module};
    use burn::record::Recorder;
    let record = trained_model.into_record();
    let type_vocab_size = 1 + parser::behavior::TYPE_DECL_LENGTH;
    let action_space = rl::action::ActionSpace::new(behavior, available_funcs);
    action_space.print_action_space();
    let action_vocab_size = action_space.size();

    let device = Default::default();
    let config = rl::model::ModelSize::Small.get_config(type_vocab_size, action_vocab_size);

    use burn::record::{BinBytesRecorder, FullPrecisionSettings};
    let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(record, ())
        .unwrap();
    let rl_record: rl::model::TransformerModelRecord<B> =
        BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes, &device)
            .unwrap();

    let rl_model = config.init::<B>(&device).load_record(rl_record);

    let target_epochs = 1000;
    let rl_trained = rl::rl::train_rl(
        rl_model,
        behavior,
        available_funcs,
        &mut runtime,
        call_args.clone(),
        target_epochs, // epochs
        32,            // batch_size
        1e-4,          // lr
        0.05,          // lambda_complexity (parsimony pressure)
        0.02,          // entropy_weight (exploration bonus)
    );

    let final_model = rl_trained.valid();

    println!("--- Saving Trained Model Weights ---");
    let record_to_save = final_model.clone().into_record();
    let model_path = format!("{}_weights.bin", behavior.name);
    burn::record::BinFileRecorder::<burn::record::FullPrecisionSettings>::default()
        .record(record_to_save, model_path.into())
        .expect("Failed to save model weights");

    for _ in 0..10 {
        let temperature: f64 = 1.0;
        let generated_sequence =
            rl::supervised::generate(behavior, available_funcs, &final_model, temperature);

        // 6) Calculate fitness
        println!("Call Args: {:?}", call_args);
        println!("Generated Sequence: {:?}", generated_sequence);

        match runtime.evaluate_sequence(&generated_sequence, call_args.clone()) {
            Ok(stdlib::Signal::DataFrame(output)) => {
                let fitness =
                    runtime::stats::evaluate_fitness_batch_add_value(&mut runtime.dmgr, &[&output]);
                println!("Inference Sample Fitness = {:?}", fitness);
                break;
            }
            _ => {
                println!("Inference Sample Runtime Execution Failed.");
            }
        };
    }
    println!("");
    println!("");
    println!("");

    println!(
        "Cache hit rate: {} / {}",
        runtime.expr_hits, runtime.expr_lookups
    );
}
