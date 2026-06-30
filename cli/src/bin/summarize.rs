#[path = "../weights.rs"]
mod weights; // cargo run --bin summarize -- --cuda --file examples/wrds_1.cm

use clap::Parser;
use parser::ast::NodeType;
use parser::behavior::BehaviorDecl;
use rl::action::ActionSpace;
use rl::model::AgentModel;
use std::fs;
use tch::nn::Module;
use tch::{Device, Tensor};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,
    #[arg(short, long)]
    cuda: bool,
}

fn main() {
    let args = Args::parse();
    let use_cuda = args.cuda || std::env::var("CUDA_PATH").is_ok();
    let device = if use_cuda {
        Device::cuda_if_available()
    } else {
        Device::Cpu
    };

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
    let weights_path = behavior_decl.weights.clone();

    let mut vs = tch::nn::VarStore::new(device);
    weights::load(&mut vs, &weights_path);
    let mut model = AgentModel::new(&vs.root(), action_space.clone(), 256);

    let seq_len = 32;
    let mut state = rl::state::SearchState::new(&network);

    let vocab_size = action_space.size();

    // 1. State-action selection distribution
    let episodes = 200;
    let mut counts = vec![vec![0; vocab_size]; vocab_size];

    let alpha_matrix = Tensor::zeros(
        [stdlib::types::SIZE[0] as i64, stdlib::types::SIZE[1] as i64],
        (tch::Kind::Float, device),
    )
    .unsqueeze(0);

    println!(
        "\n=== Running {} episodes for state-action distribution ===",
        episodes
    );
    for _ep in 0..episodes {
        state.reset();
        let mut actions = Vec::new();
        let mut last_action = 0; // SOS

        for _step in 0..seq_len {
            let (stack, _callgraph) = state.machine.get_stack();
            let mut valid_actions = vec![];
            for action_idx in 0..action_space.size() {
                let action = action_space.get_action(action_idx);
                let valid = match &action {
                    rl::action::Action::Done => {
                        stack.len() == 1
                            && matches!(stack[0].0, stdlib::types::Signal::DataFrame(_))
                    }
                    rl::action::Action::Reduce(op_spec) => {
                        stack.len() >= op_spec.inputs.len() && state.machine.check_reduce(&op_spec)
                    }
                    _ => true,
                };
                if valid {
                    valid_actions.push(action);
                }
            }
            let mask = action_space.calculate_mask(&valid_actions).to(device);

            let mut target_tokens = actions.clone();
            target_tokens.insert(0, 0); // SOS
            target_tokens.truncate(seq_len as usize);
            while target_tokens.len() < seq_len as usize {
                target_tokens.push(0);
            }
            let shifted_target_tokens = Tensor::from_slice(&target_tokens).to(device).unsqueeze(0);

            let logits =
                tch::no_grad(|| model.decoder.forward(&shifted_target_tokens, &alpha_matrix));
            let step_idx = std::cmp::min(actions.len(), seq_len as usize - 1) as i64;
            let action_logits = logits.select(1, step_idx);
            let action_logits =
                action_logits.masked_fill(&mask.logical_not().unsqueeze(0), std::f64::NEG_INFINITY);

            let log_probs = action_logits.log_softmax(-1, tch::Kind::Float);
            let mut probs = log_probs.exp().nan_to_num(0.0, 0.0, 0.0).clamp(0.0, 1.0);
            let sum = probs.sum(tch::Kind::Float);
            if sum.double_value(&[]) <= 1e-8 {
                probs = mask.unsqueeze(0).to_kind(tch::Kind::Float);
                let mask_sum = probs.sum(tch::Kind::Float);
                probs = &probs / mask_sum;
            } else {
                probs = &probs / &sum;
            }

            let sampled_action_idx: Vec<Vec<i64>> = tch::no_grad(|| probs.multinomial(1, true))
                .try_into()
                .unwrap();
            let action_idx = sampled_action_idx[0][0] as usize;

            counts[action_idx][last_action] += 1;
            last_action = action_idx;

            let action = action_space.get_action(action_idx);
            state.apply_action(&action);
            actions.push(action_idx as i64);

            if action == rl::action::Action::Done {
                break;
            }
        }
    }

    println!("\n=== Action Index Map ===");
    for i in 0..vocab_size {
        let act = action_space.get_action(i);
        let act_str: String = (&act).into();
        println!("{:>3}: {}", i, act_str);
    }

    println!("\n=== Transition Probability Matrix ===");
    println!("Rows: Next Action | Columns: Last Action");
    print!("{:>4} |", "");
    for col in 0..vocab_size {
        let col_sum: i32 = (0..vocab_size).map(|r| counts[r][col]).sum();
        if col_sum > 0 {
            print!("{:>5} ", col);
        }
    }
    println!();
    println!("{:-<100}", "");

    for row in 0..vocab_size {
        let row_sum: i32 = counts[row].iter().sum();
        if row_sum == 0 {
            continue;
        }
        print!("{:>4} |", row);
        for col in 0..vocab_size {
            let col_sum: i32 = (0..vocab_size).map(|r| counts[r][col]).sum();
            if col_sum > 0 {
                let prob = counts[row][col] as f64 / col_sum as f64;
                if prob > 0.005 {
                    print!("{:>5.2} ", prob);
                } else {
                    print!("{:>5} ", "-");
                }
            }
        }
        println!();
    }

    // 2. Value network visualization
    println!(
        "\n=== Value Network Visualization (\u{03bc} and \u{03c3} over varying complexity) ==="
    );
    println!("Evaluating with fixed data & ops ratio, varying sequence length (complexity).");
    for len in 1..=30 {
        let mut history = vec![];
        for _ in 0..len {
            history.push(1); // dummy action to construct 5d embedding
        }
        let emb_5d = AgentModel::compute_5d_embedding(&history, false, &action_space);
        let emb_5d_t = Tensor::from_slice(&emb_5d).to(device).unsqueeze(0);

        let value_out = tch::no_grad(|| model.value_net.forward(&emb_5d_t).squeeze_dim(0));
        let mu = value_out.double_value(&[0]);
        let log_sigma = value_out.double_value(&[1]).clamp(-20.0, 2.0);
        let sigma = std::f64::consts::E.powf(log_sigma);

        println!(
            "Length (Complexity): {:>2} | \u{03bc} = {:>7.4} | \u{03c3} = {:>7.4}",
            len, mu, sigma
        );
    }
}
