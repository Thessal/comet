use clap::Parser;
use parser::parser::parse;
use parser::program::{BehaviorDecl, Declaration};
use std::fs;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input .cm file to compile
    file: String,
}

fn generate_valid_expressions(
    behavior: &BehaviorDecl,
    available_funcs: &Vec<(
        String,
        Vec<parser::program::TypeDecl>,
        parser::program::TypeDecl,
    )>,
    k: usize,
    runtime: &mut runtime::runtime::Runtime,
) -> Vec<rl::search::EvaluatedSample> {
    // TODO: generate top k and drop duplicate samples. the result count may be smaller than k
    println!("Inferring action candidates from the code:");
    for f in available_funcs {
        println!("- {}({:?}) -> {:?}", f.0, f.1, f.2);
    }

    println!("\nGenerating sample expression trees and evaluating using runtime...");
    let samples = rl::search::generate_top_k_samples(
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
            available_funcs.push(rl::env::get_available_func(&func_str));
        }
        available_funcs
    } else {
        rl::env::get_available_funcs()
    };
    println!("--- Available functions : {:?} ---", available_funcs);

    // Initialize central runtime
    let mut runtime = runtime::runtime::Runtime::new(100, "data");

    // type BackendAutoDiff = burn::backend::Autodiff<burn::backend::Cuda>;
    // type BackendAutoDiff = burn::backend::Autodiff<burn::backend::Rocm>;
    type BackendAutoDiff = burn::backend::Autodiff<burn::backend::ndarray::NdArray>;

    // 1) Build dataset, 2) Train transformer,
    let sample = generate_valid_expressions(&behavior, &available_funcs, 100, &mut runtime);
    let trained_model = rl::supervised::train::<BackendAutoDiff>(
        &behavior,
        &available_funcs,
        &sample,
        behavior.supervised_samples.unwrap(),
        32, //bs
        16, //num_worker
    );
    let _ = std::io::stdout().flush();
    println!(
        "Cache hit rate: {} / {}",
        runtime.expr_hits, runtime.expr_lookups
    );

    // 3) Sample a behavior using trained transformer
    // 4) Evaluate the polish sequence using runtime ## TODO: wrap this, and write a test for this.
    // Extract bound parameter values from the Flow call syntax
    let mut call_args = behavior
        .args
        .iter()
        .map(|_arg| "volume".to_string())
        .collect::<Vec<_>>();

    for decl in &program.declarations {
        if let Declaration::Flow(f) = decl {
            for stmt in &f.body {
                if let parser::program::FlowStmt::Expr(parser::program::Expr::Call { path, args }) =
                    stmt
                {
                    if path.segments.join("::") == behavior.name {
                        let mut p = Vec::new();
                        for arg in args {
                            match arg {
                                parser::program::Expr::Identifier(name) => p.push(name.clone()),
                                parser::program::Expr::Literal(
                                    parser::program::Literal::Float(f),
                                ) => p.push(f.to_string()),
                                parser::program::Expr::Literal(
                                    parser::program::Literal::Integer(i),
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

    // -- RL Fine-Tuning Step --
    println!("--- Starting RL Fine-Tuning ---");
    use burn::module::{AutodiffModule, Module};
    use burn::record::Recorder;
    let record = trained_model.into_record();
    let type_vocab_size = 1 + parser::program::TYPE_DECL_LENGTH;
    let action_vocab_size = 2 // Done, Shift
        + behavior.floats.as_ref().map_or(0, |v| v.len())
        + behavior.integers.as_ref().map_or(0, |v| v.len())
        + behavior.strings.as_ref().map_or(0, |v| v.len())
        + available_funcs.len();

    let device = Default::default();
    let config = rl::model::ModelSize::Small.get_config(type_vocab_size, action_vocab_size);

    use burn::record::{BinBytesRecorder, FullPrecisionSettings};
    let bytes = BinBytesRecorder::<FullPrecisionSettings>::default()
        .record(record, ())
        .unwrap();
    let rl_record: rl::model::TransformerModelRecord<BackendAutoDiff> =
        BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(bytes, &device)
            .unwrap();

    let rl_model = config
        .init::<BackendAutoDiff>(&device)
        .load_record(rl_record);

    let mut eval_fn = |sequence: &[String]| -> f64 {
        match runtime.evaluate_sequence(sequence, call_args.clone()) {
            Ok(stdlib::ParamType::DataFrame(output)) => {
                let fitness = runtime::fitness::evaluate_fitness(&mut runtime.dmgr, &output);
                // Multi-objective fitness returns a vector; use the first metric
                100. * fitness.first().copied().unwrap_or(0.0)
            }
            _ => 0.0, // Failed runtime evaluations score 0
        }
    };

    let target_epochs = 100;
    let rl_trained = rl::rl::train_rl(
        rl_model,
        &behavior,
        &available_funcs,
        eval_fn,
        target_epochs, // epochs
        32,            // batch_size
        1e-3,          // lr
        0.05,          // lambda_complexity (parsimony pressure)
        0.02,          // entropy_weight (exploration bonus)
    );

    let final_model = rl_trained.valid();

    for _ in 0..10 {
        let temperature: f64 = 1.0;
        let generated_sequence =
            rl::supervised::generate(&behavior, &available_funcs, &final_model, temperature);

        // 6) Calculate fitness
        println!("Call Args: {:?}", call_args);
        println!("Generated Sequence: {:?}", generated_sequence);

        match runtime.evaluate_sequence(&generated_sequence, call_args.clone()) {
            Ok(stdlib::ParamType::DataFrame(output)) => {
                let fitness = runtime::fitness::evaluate_fitness(&mut runtime.dmgr, &output);
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
