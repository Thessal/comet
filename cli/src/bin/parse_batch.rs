//   ### Usage Instructions

//   Run the command below from the workspace root or cli/ folder:

//     cargo run --bin parse_batch -- \
//       --behavior-file examples/mixed_4.cm \
//       --equations-file equations.txt \
//       --output-file returns_parsed.csv


use clap::Parser;
use parser::ast::NodeType;
use parser::behavior::BehaviorDecl;
use runtime::backtest::BasicBacktest;
use runtime::runtime::Runtime;
use runtime::stats::MetricChoice;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use stdlib::types::Signal;
use tch::Device;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "examples/wrds_1.cm")]
    behavior_file: String,
    #[arg(short, long, default_value = "equations.txt")]
    equations_file: String,
    #[arg(short, long, default_value = "returns_parsed.csv")]
    output_file: String,
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

    println!("--- Reading behavior file: {:?} ---", args.behavior_file);
    let src = fs::read_to_string(&args.behavior_file)
        .unwrap_or_else(|_| panic!("Failed to read behavior file: {}", args.behavior_file));

    let (network, behavior_nodes) = parser::parser::parse(&src)
        .unwrap_or_else(|e| panic!("Failed to parse behavior file {:?}: {}", args.behavior_file, e));

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => panic!("Expected behavior node"),
    };

    let datadir = behavior_decl
        .datadir
        .clone()
        .unwrap_or_else(|| "./data".to_string());

    let info_str = fs::read_to_string(format!("{}/info.json", datadir))
        .unwrap_or_else(|_| panic!("Failed to read {}/info.json", datadir));
    let info: serde_json::Value =
        serde_json::from_str(&info_str).expect("Failed to parse info.json");
    let target_field = info["target"].as_str().expect("Target field missing in info.json");

    stdlib::types::set_size(
        info["rows"].as_i64().expect("rows missing in info.json"),
        info["columns"].as_i64().expect("columns missing in info.json"),
    );

    let metric_str = behavior_decl
        .metric
        .clone()
        .unwrap_or_else(|| "sharpe".to_string());
    let metric: MetricChoice = metric_str.into();

    let mut runtime = Runtime::new(1000, datadir.into(), Some(device));
    let backtester = BasicBacktest::new(&mut runtime.dmgr, target_field, metric);

    println!("--- Reading equations file: {:?} ---", args.equations_file);
    let file = File::open(&args.equations_file)
        .unwrap_or_else(|_| panic!("Failed to open equations file: {}", args.equations_file));
    let reader = BufReader::new(file);

    let mut out_file = File::create(&args.output_file)
        .unwrap_or_else(|_| panic!("Failed to create output CSV file: {}", args.output_file));

    let mut count = 0;
    for line in reader.lines() {
        let line = line.expect("Failed to read line from equations file");
        let expr_str = line.trim();
        if expr_str.is_empty() || expr_str.starts_with('#') {
            continue;
        }

        let flow_src = format!(
            "Behavior Placeholder(d: DataFrame) {{}} -> DataFrame\nFlow f {{\n  {}\n}}\n",
            expr_str
        );

        let (eq_network, _) = match parser::parser::parse(&flow_src) {
            Ok(res) => res,
            Err(e) => {
                eprintln!("Failed to parse expression '{}': {}", expr_str, e);
                continue;
            }
        };

        let signal = runtime.lookup_or_run(&eq_network, eq_network.root);

        let returns_tensor = match signal {
            Signal::DataFrame(Some(tensor)) => backtester.calc_returns(tensor),
            _ => {
                eprintln!("Expression '{}' did not evaluate to a DataFrame", expr_str);
                continue;
            }
        };

        let vals = Vec::<f64>::try_from(returns_tensor.contiguous().to_kind(tch::Kind::Double))
            .expect("Failed to convert returns tensor to Vec<f64>");
        let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();

        let expr_escaped = expr_str.replace("\"", "\"\"");
        let csv_line = format!("\"{}\",{}\n", expr_escaped, vals_str.join(","));
        out_file
            .write_all(csv_line.as_bytes())
            .expect("Failed to write to output CSV file");
        count += 1;
    }

    println!(
        "Successfully evaluated {} equations and wrote returns sequence to {:?}",
        count, args.output_file
    );
}
