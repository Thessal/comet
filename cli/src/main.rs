use clap::Parser;
use parser::behavior::InputDecl;
use parser::expr::{Expr, Literal};
use parser::parser::parse;
use std::collections::HashMap;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input .cm file to compile
    file: String,

    /// Optional flag to use CUDA backend
    #[arg(long)]
    cuda: bool,
}

fn build_data_param(name: String) -> runtime::ast::Program {
    runtime::ast::Program {
        spec: runtime::ast::OperatorSpec {
            name: "data".to_string(),
            inputs_type: vec![stdlib::types::Signal::String(None)],
            output_type: stdlib::types::Signal::DataFrame(None),
        },
        polish_expression: Some(vec![
            runtime::ast::Token::Literal(parser::expr::Literal::String(name.clone())),
            runtime::ast::Token::Operator("data".into()),
        ]),
        parameters: Some(vec![runtime::ast::Tree::Literal(
            parser::expr::Literal::String(name),
        )]),
    }
}

fn main() {
    let args = Args::parse();
    _main(args);
}
fn _main(args: Args) {
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let program = parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    // Select first train=True Behavior
    let mut target_behavior = None;
    for decl in &program {
        if let InputDecl::Behavior(b) = decl {
            if Some(true) == b.train {
                target_behavior = Some(b.clone());
            }
        }
    }
    let behavior = target_behavior.expect("No train=True behavior found");
    println!("--- Selected behavior ---");

    // Initialize central runtime
    let mut runtime = runtime::runtime::Runtime::new(10000, "data".into());

    // Extract bound parameter values from the Flow call syntax
    let mut call_args = behavior
        .inputs
        .iter()
        .map(|_arg| "volume".to_string())
        .collect::<Vec<_>>();

    for decl in &program {
        if let InputDecl::Flow(f) = decl {
            for stmt in &f.body {
                if let parser::expr::Stmt::Expr(Expr::Call { args, .. }) = stmt {
                    if args.len() == behavior.inputs.len() {
                        let mut p = Vec::new();
                        for arg in args {
                            match arg {
                                Expr::Identifier(name) => p.push(name.clone()),
                                Expr::Literal(Literal::Float(f)) => p.push(f.to_string()),
                                Expr::Literal(Literal::Integer(i)) => p.push(i.to_string()),
                                _ => p.push("volume".to_string()),
                            }
                        }
                        call_args = p;
                    }
                }
            }
        }
    }

    call_args.reverse();

    let params: Vec<runtime::ast::Program> = call_args
        .into_iter()
        .map(|arg| build_data_param(arg))
        .collect();

    let device = if use_cuda {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };

    println!("--- Starting RL Fine-Tuning ---");
    let score_fn = runtime::stats::Aggregator {
        weights: HashMap::from_iter([
            (runtime::stats::Metric::Sharpe, (0.5, 0., 1.)),
            (runtime::stats::Metric::Ret, (0.5, 0., 1.)),
        ]),
    };

    let mut env = rl::env::Environment::new(
        &mut runtime,
        behavior.clone(),
        params,
        score_fn,
        10, // max_length
        32, // batch_size
    );

    let action_vocab_size = env.action_space.size();
    let config =
        rl::model::ModelConfig::RnnModel(rl::model::ModelSize::Small.get_config(action_vocab_size));
    let vs = tch::nn::VarStore::new(device);
    let model = config.init(&vs.root());

    env.run(&model, device);

    println!("--- Evaluation ---");
    for i in 0..10 {
        let traj = env.sample_trajectory(&model, device);
        println!("Sample {}:", i);
        for step in traj {
            println!("  Action: {:?}", step.action);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::_main;
    use crate::Args;

    #[test]
    fn test_example_1() {
        _main(Args {
            file: "../examples/behavior_1.cm".to_string(),
            cuda: false,
        })
    }
}
