use clap::Parser;
use parser::behavior::InputDecl;
use parser::expr::{Expr, Literal};
use parser::parser::Rule::behavior_decl;
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
    runtime::ast::Program::new(
        "data",
        vec![runtime::ast::Tree::Literal(parser::expr::Literal::String(
            name,
        ))],
    )
}

fn brute_force(ast: ast::Program, behavior_decls: Vec<behavior_decl>) {
    // 1. build environment from ast, behavior_decls. (rl::env::Environment::new and rl::env::Environment have to updated)
    // 2. sample valid actions, randomly
    // 3. store results that are valid and finished successfully.
    // 4. repelat until it generates 3 different results.
}

//TODO: machine generated code. need verification.
fn main() {
    let args = Args::parse();
    _main(args);
}
fn _main(args: Args) {
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let (tree, behavior_decls) =
        parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    brute_force(tree, behavior_decl);

    // // Select first train=True Behavior
    // let mut target_behavior = None;
    // for decl in &behavior_decls {
    //     if let InputDecl::Behavior(b) = decl {
    //         if Some(true) == b.train {
    //             target_behavior = Some(b.clone());
    //         }
    //     }
    // }
    // let behavior = target_behavior.expect("No train=True behavior found");
    // println!("--- Selected behavior ---");

    // // Initialize central runtime
    // let mut runtime = runtime::runtime::Runtime::new(10000, "data".into(), None);
    // runtime.enable = false; // NOTE: dummy runtime

    // let device = if use_cuda {
    //     tch::Device::Cuda(0)
    // } else {
    //     tch::Device::Cpu
    // };

    // println!("--- Starting RL Fine-Tuning ---");
    // let score_fn = runtime::stats::Aggregator {
    //     weights: HashMap::from_iter([
    //         // (runtime::stats::Metric::Sharpe, (0.5, 0., 1.)),
    //         // (runtime::stats::Metric::Ret, (0.5, 0., 1.)),
    //     ]),
    // };

    // let mut env = rl::env::Environment::new(
    //     &mut runtime,
    //     behavior.clone(),
    //     params,
    //     score_fn,
    //     20,   // max_length
    //     1024, // batch_size
    // );

    // let action_vocab_size = env.action_space.size();
    // let config =
    //     rl::model::ModelConfig::RnnModel(rl::model::ModelSize::Small.get_config(action_vocab_size));
    // let vs = tch::nn::VarStore::new(device);
    // let model = config.init(&vs.root());

    // env.run(&model, 100, device);

    // println!("--- Evaluation ---");
    // for i in 0..10 {
    //     let traj = env.sample_trajectory(&model, device);
    //     println!("Sample {}:", i);
    //     for step in traj {
    //         println!("  Action: {:?}", step.action);
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use crate::_main;
    use crate::Args;

    #[test]
    fn test_behavior_1() {
        _main(Args {
            file: "../examples/behavior_1.cm".to_string(),
            cuda: false,
        })
    }
}
