#[path = "../weights.rs"]
mod weights;
// cargo run --bin batch_summarize -- --cuda --file examples/wrds_1.cm
use clap::Parser;
use parser::ast::NodeType;
use parser::behavior::BehaviorDecl;
use rl::action::ActionSpace;
use rl::model::AgentModel;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
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

fn find_weight_files(weights_base: &str) -> Vec<String> {
    let mut files = Vec::new();

    // 1. Check for indexed files weights_base.0, weights_base.1, ...
    let mut idx = 0;
    loop {
        let candidate = format!("{}.{}", weights_base, idx);
        if Path::new(&candidate).exists() {
            files.push(candidate);
            idx += 1;
        } else {
            break;
        }
    }

    // 2. If no indexed files found via loop, search directory for matches
    if files.is_empty() {
        let base_path = Path::new(weights_base);
        if let Some(parent) = base_path.parent() {
            let dir_to_search = if parent.as_os_str().is_empty() {
                Path::new(".")
            } else {
                parent
            };

            if let Ok(entries) = fs::read_dir(dir_to_search) {
                let file_stem = base_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy();
                let prefix = format!("{}.", file_stem);

                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                            if name.starts_with(&prefix) && !name.ends_with(".csv") {
                                files.push(path.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Fallback: check if the exact base path exists
    if files.is_empty() && Path::new(weights_base).exists() {
        files.push(weights_base.to_string());
    }

    files.sort();
    files
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

    println!("--- Parsing behavior specification file: {:?} ---", filename);
    let (network, behavior_nodes) =
        parser::parser::parse(&src).expect(format!("Failed to parse {:?}", filename).as_str());

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => unreachable!(),
    };
    let action_space: ActionSpace = behavior_decl.into();

    let default_weights_base = format!(
        "weights/{}.pth",
        Path::new(filename)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    );

    let weights_base = behavior_decl
        .weights
        .clone()
        .unwrap_or(default_weights_base);

    let weight_files = find_weight_files(&weights_base);

    if weight_files.is_empty() {
        eprintln!(
            "No weight checkpoint files found matching pattern '{}.*' or '{}'",
            weights_base, weights_base
        );
        return;
    }

    println!(
        "\nFound {} weight file(s) for pattern '{}':",
        weight_files.len(),
        weights_base
    );
    for w in &weight_files {
        println!("  - {}", w);
    }

    for (agent_idx, weight_file) in weight_files.iter().enumerate() {
        println!(
            "\n=== Processing Weight File [{}/{}]: {} ===",
            agent_idx + 1,
            weight_files.len(),
            weight_file
        );

        let mut vs = tch::nn::VarStore::new(device);
        weights::load(&mut vs, &Some(weight_file.clone()));
        let mut model = AgentModel::new(&vs.root(), action_space.clone(), 256);

        let csv_filename = format!("{}.summary.csv", weight_file);
        let mut file = fs::File::create(&csv_filename).expect("Failed to create CSV file");

        let header = "length,num_data,num_ops,entropy,is_done,mu,sigma";
        writeln!(file, "{}", header).unwrap();

        println!("Generating Value Network Grid Scan -> {}", csv_filename);

        for len in 1..=30 {
            for num_data in 1..=10 {
                for num_ops in 1..=10 {
                    for entropy_idx in 0..=30 {
                        let entropy = (entropy_idx as f32) * 0.1;
                        for &is_done_bool in &[false, true] {
                            let is_done = if is_done_bool { 1.0 } else { 0.0 };
                            let scale = 30.0;
                            let emb_5d = [
                                (len as f32) / scale,
                                (num_data as f32) / scale,
                                (num_ops as f32) / scale,
                                entropy,
                                is_done,
                            ];
                            let emb_5d_t = Tensor::from_slice(&emb_5d).to(device).unsqueeze(0);

                            let value_out =
                                tch::no_grad(|| model.value_net.forward(&emb_5d_t).squeeze_dim(0));
                            let mu = value_out.double_value(&[0]);
                            let log_sigma = value_out.double_value(&[1]).clamp(-20.0, 2.0);
                            let sigma = std::f64::consts::E.powf(log_sigma);

                            let line = format!(
                                "{},{},{},{:.1},{},{:.6},{:.6}",
                                len, num_data, num_ops, entropy, is_done, mu, sigma
                            );
                            writeln!(file, "{}", line).unwrap();
                        }
                    }
                }
            }
        }
        println!("Successfully saved pareto curve data to {}", csv_filename);
    }

    println!(
        "\n=== Batch summary complete for {} weight file(s) ===",
        weight_files.len()
    );
}
