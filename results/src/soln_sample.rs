use parser::parser::parse;
use parser::program::Declaration;
use runtime::backtest::mockbacktester;
use std::fs;
use std::path::{Path, PathBuf};

fn evaluate_samples(
    samples_seqs: Vec<Vec<String>>,
    call_args: Vec<String>,
    runtime: &mut runtime::runtime::Runtime,
) -> Vec<(Vec<String>, Vec<f64>, Vec<f64>)> {
    let mut results = Vec::new();
    for seq in samples_seqs {
        match runtime.evaluate_sequence(&seq, call_args.clone()) {
            Ok(stdlib::ParamType::DataFrame(output)) => {
                let pnl = runtime::backtest::mockbacktester(&mut runtime.dmgr, &output);
                let fitness = runtime::fitness::evaluate_fitness(&mut runtime.dmgr, &output);
                results.push((seq, fitness, pnl));
            }
            _ => {
                // Ignore parsing or execution errors on sequences
            }
        }
    }
    results
}

fn write_json(path: &Path, results: &Vec<(Vec<String>, Vec<f64>, Vec<f64>)>) {
    let mut file = fs::File::create(path).unwrap();
    use std::io::Write;
    write!(file, "[\n").unwrap();
    for (i, (seq, fit, pnl)) in results.iter().enumerate() {
        write!(file, "  {{\n").unwrap();

        // Write sequence
        let seq_str = seq
            .iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>()
            .join(", ");
        write!(file, "    \"sequence\": [{}],\n", seq_str).unwrap();

        // Write fitness
        let fit_str = fit
            .iter()
            .map(|f| format!("{}", f))
            .collect::<Vec<_>>()
            .join(", ");
        write!(file, "    \"fitness\": [{}],\n", fit_str).unwrap();

        // Write correlation series (mock PnL array)
        let series = pnl
            .iter()
            .map(|v| {
                if v.is_nan() {
                    "null".to_string()
                } else {
                    format!("{}", v)
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        write!(file, "    \"series\": [{}]\n", series).unwrap();

        if i == results.len() - 1 {
            write!(file, "  }}\n").unwrap();
        } else {
            write!(file, "  }},\n").unwrap();
        }
    }
    write!(file, "]\n").unwrap();
}

fn main() {
    // Expect to be run from `comet/` directory OR `comet/results/`
    let base_dir = if Path::new("results").exists() {
        PathBuf::from("results")
    } else {
        PathBuf::from(".")
    };

    let data_dir = if Path::new("data/volume.csv").exists() {
        "data"
    } else {
        "../data"
    };

    println!("Scanning trials in {:?}", base_dir);
    let dirs = fs::read_dir(&base_dir).unwrap();
    let mut trials = Vec::new();
    for entry in dirs {
        let entry = entry.unwrap();
        let name = entry.file_name().into_string().unwrap();
        if name.starts_with("_") && entry.path().is_dir() {
            trials.push(entry.path());
        }
    }
    trials.sort();

    for trial_dir in trials {
        println!("--------------------------------------------------");
        println!("Processing trial {:?}", trial_dir);
        let cm_path = trial_dir.join("behavior_1.cm");
        if !cm_path.exists() {
            println!("No behavior_1.cm found, skipping.");
            continue;
        }

        let src = fs::read_to_string(&cm_path).unwrap();
        let program = parse(&src).unwrap();
        let mut target_behavior = None;
        let mut call_args = Vec::new();

        for decl in &program.declarations {
            if let Declaration::Behavior(b) = decl {
                if Some(true) == b.train {
                    target_behavior = Some(b.clone());
                }
            }
            if let Declaration::Flow(f) = decl {
                for stmt in &f.body {
                    if let parser::program::FlowStmt::Expr(parser::program::Expr::Call {
                        path,
                        args,
                    }) = stmt
                    {
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
                                _ => p.push("volume".to_string()), // fallback
                            }
                        }
                        call_args = p;
                    }
                }
            }
        }

        let behavior = target_behavior.unwrap();
        // `evaluate_sequence` pops from end, so reverse expected
        call_args.reverse();

        let available_funcs = if let Some(available_funcs_str) = &behavior.operators {
            let mut funcs = Vec::new();
            for func_str in available_funcs_str {
                funcs.push(rl::env::get_available_func(func_str));
            }
            funcs
        } else {
            rl::env::get_available_funcs()
        };

        let mut runtime = runtime::runtime::Runtime::new(100000, data_dir);

        // Model loading
        type BackendAutoDiff = burn::backend::Autodiff<burn::backend::NdArray>;
        let device = Default::default();
        let type_vocab_size = 1 + parser::program::TYPE_DECL_LENGTH;
        let action_vocab_size = 2 // Done, Shift
            + behavior.floats.as_ref().map_or(0, |v| v.len())
            + behavior.integers.as_ref().map_or(0, |v| v.len())
            + behavior.strings.as_ref().map_or(0, |v| v.len())
            + available_funcs.len();

        let config = rl::model::ModelSize::Small.get_config(type_vocab_size, action_vocab_size);
        let model_path = trial_dir.join(format!("{}_weights.bin", behavior.name));

        let mut model_samples_seqs = Vec::new();

        if model_path.exists() {
            println!("Loading model weights from {:?}", model_path);
            let model_bytes =
                fs::read(&model_path).expect("Should be able to read model weights file");

            use burn::module::Module;
            use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
            let rl_record: rl::model::TransformerModelRecord<BackendAutoDiff> =
                BinBytesRecorder::<FullPrecisionSettings>::default()
                    .load(model_bytes, &device)
                    .expect("Should be able to load model the model weights from bytes");
            let rl_model = config
                .init::<BackendAutoDiff>(&device)
                .load_record(rl_record);

            println!("Sampling 1000 equations from RL model...");
            for _ in 0..1000 {
                let generated =
                    rl::supervised::generate(&behavior, &available_funcs, &rl_model, 1.0);
                model_samples_seqs.push(generated);
            }
        } else {
            println!("Model weights NOT found at {:?}", model_path);
        }

        println!("Evaluating Model samples...");
        let model_results = evaluate_samples(model_samples_seqs, call_args.clone(), &mut runtime);
        println!("Model yields {} valid executions", model_results.len());
        let out_model = trial_dir.join("soln_model_1000.json");
        write_json(&out_model, &model_results);

        println!("Sampling 1000 equations from Random Agent via generate_top_k_samples...");
        // generate_top_k_samples returns distinct valid samples
        let random_evaluated = rl::search::generate_top_k_samples(
            &behavior,
            &available_funcs,
            1000,
            |fitness| fitness.first().copied().unwrap_or(0.0) != -99999.0, // Accept any valid parsed output
            &mut runtime,
        );

        let random_samples_seqs = random_evaluated
            .into_iter()
            .map(|s| s.actions)
            .collect::<Vec<_>>();
        println!("Evaluating Random samples...");
        let random_results = evaluate_samples(random_samples_seqs, call_args.clone(), &mut runtime);
        println!("Random yields {} valid executions", random_results.len());
        let out_random = trial_dir.join("soln_random_1000.json");
        write_json(&out_random, &random_results);
    }
}
