pub mod bruteforce;
pub mod transformer;
use clap::Parser;
use parser::ast::NodeType;
use parser::behavior::BehaviorDecl;
use rl::action::ActionSpace;
use std::fs;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,
    #[arg(short, long)]
    cuda: bool,
}

fn main() {
    _main(Args::parse());
}

fn _main(args: Args) {
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let (network, behavior_nodes) =
        parser::parser::parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => unreachable!(),
    };
    let action_space: ActionSpace = behavior_decl.into();
    // bruteforce::brute_force(network, action_space, use_cuda);
    let pool = transformer::transformer_search(network, action_space, use_cuda, None);

    println!("--- Expressions found ---");
    for expr in pool.exprs() {
        println!("{}", expr);
    }

    pool.save_returns("returns.csv")
}

fn _main_bruteforce(args: Args) {
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let (network, behavior_nodes) =
        parser::parser::parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => unreachable!(),
    };
    let action_space: ActionSpace = behavior_decl.into();
    let pool = bruteforce::brute_force(network, action_space, use_cuda);

    println!("--- Expressions found ---");
    for expr in pool.exprs() {
        println!("{}", expr);
    }

    pool.save_returns("returns_brute.csv")
}

fn _main_standard_ppo(args: Args) {
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let filename = &args.file;
    let src = fs::read_to_string(filename).expect("Failed to read file");

    println!("--- Parsing file: {:?} ---", filename);
    let (network, behavior_nodes) =
        parser::parser::parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => unreachable!(),
    };
    let action_space: ActionSpace = behavior_decl.into();
    // bruteforce::brute_force(network, action_space, use_cuda);
    let pool = transformer::transformer_search(network, action_space, use_cuda, Some(0.0));

    println!("--- Expressions found ---");
    for expr in pool.exprs() {
        println!("{}", expr);
    }

    pool.save_returns("returns_ppo.csv")
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_behavior_1() {
        let filename = "../examples/behavior_2.cm";
        let _src = fs::read_to_string(filename).expect("Failed to read file");
        _main_bruteforce(Args {
            file: String::from(filename),
            cuda: true,
        });
    }
    #[test]
    fn test_behavior_ppo() {
        let filename = "../examples/behavior_2.cm";
        let _src = fs::read_to_string(filename).expect("Failed to read file");
        _main_standard_ppo(Args {
            file: String::from(filename),
            cuda: true,
        });
    }
}
