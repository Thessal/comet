pub mod bruteforce;
pub mod search;
use clap::Parser;
use parser::ast::{Network, Node, NodeType};
use parser::behavior::BehaviorDecl;
use parser::behavior::InputDecl;
use parser::expr::{Expr, Literal};
use std::collections::HashMap;
use std::fs;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,
    #[arg(short, long)]
    cuda: bool,
}

//TODO: machine generated code. need verification.
// fn main() {
//     let args = Args::parse();
//     _main(args);
// }
fn main() {
    let args = Args::parse();
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let (network, root, behavior_nodes) =
        parser::parser::parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    let behavior_decls = Vec::new(); // Dummy since it was undefined
    let action_space = rl::action::ActionSpace::new();
    bruteforce::brute_force(network, root, action_space, behavior_decls, behavior_nodes);

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

// #[cfg(test)]
// mod tests {
//     use crate::_main;
//     use crate::Args;

//     #[test]
//     fn test_behavior_1() {
//         _main(Args {
//             file: "../examples/behavior_1.cm".to_string(),
//             cuda: false,
//         })
//     }
// }
