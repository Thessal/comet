use clap::Parser;
use parser::parser::parse;
use parser::program::{BehaviorDecl, Declaration};
use std::fs;

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
) -> Vec<rl::search::EvaluatedSample> {
    println!("Inferring action candidates from the code:");
    for f in available_funcs {
        println!("- {}({:?}) -> {:?}", f.0, f.1, f.2);
    }

    println!("\nGenerating sample expression trees and evaluating using runtime...");
    let samples = rl::search::generate_top_k_samples(&behavior, available_funcs, 3);

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
    let available_funcs = rl::env::get_available_funcs();

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

    let sample = generate_valid_expressions(&behavior, &available_funcs);

    // 1) Build dataset, 2) Train transformer, 3) Sample a behavior using trained transformer
    let generated_sequence = rl::supervised::train_and_sample(
        &behavior,
        &available_funcs,
        &sample,
        behavior.supervised_samples.unwrap(),
    );

    // 5) Evaluate the polish sequence using runtime
    // 6) Calculate fitness
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

    let mut runtime = runtime::runtime::Runtime::new(100, "data");

    match runtime.evaluate_sequence(&generated_sequence, call_args) {
        Ok(stdlib::ParamType::DataFrame(output)) => {
            let fitness = runtime::fitness::evaluate_fitness(&mut runtime.dmgr, &output);
            println!("Inference Sample Fitness = {:?}", fitness);
        }
        _ => {
            println!("Inference Sample Runtime Execution Failed.");
        }
    };
}
