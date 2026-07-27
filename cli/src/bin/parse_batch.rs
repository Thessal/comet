//   ### Usage Instructions
//
//   Run the command below from the workspace root or cli/ folder:
//
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
    #[arg(long)]
    output_file_is: Option<String>,
    #[arg(long)]
    output_file_os: Option<String>,
    #[arg(long, default_value = "./data_is")]
    datadir_is: String,
    #[arg(long, default_value = "./data_os")]
    datadir_os: String,
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

    let metric_str = behavior_decl
        .metric
        .clone()
        .unwrap_or_else(|| "sharpe".to_string());
    let metric: MetricChoice = metric_str.into();

    // 1. Setup In-Sample Data Manager & Runtime
    let datadir_is = &args.datadir_is;
    println!("--- Setting up In-Sample Runtime using datadir: {} ---", datadir_is);
    let info_str_is = fs::read_to_string(format!("{}/info.json", datadir_is))
        .unwrap_or_else(|_| panic!("Failed to read {}/info.json", datadir_is));
    let info_is: serde_json::Value =
        serde_json::from_str(&info_str_is).expect("Failed to parse IS info.json");
    let target_field_is = info_is["target"].as_str().expect("Target field missing in IS info.json").to_string();
    let rows_is = info_is["rows"].as_i64().expect("rows missing in IS info.json");
    let cols_is = info_is["columns"].as_i64().expect("columns missing in IS info.json");

    stdlib::types::set_size(rows_is, cols_is);
    let mut runtime_is = Runtime::new(1000, datadir_is.clone().into(), Some(device));
    runtime_is.dmgr.data_size = (rows_is as usize, cols_is as usize);
    let backtester_is = BasicBacktest::new(&mut runtime_is.dmgr, &target_field_is, metric.clone());

    // 2. Setup Out-Sample Data Manager & Runtime
    let datadir_os = &args.datadir_os;
    println!("--- Setting up Out-Sample Runtime using datadir: {} ---", datadir_os);
    let info_str_os = fs::read_to_string(format!("{}/info.json", datadir_os))
        .unwrap_or_else(|_| panic!("Failed to read {}/info.json", datadir_os));
    let info_os: serde_json::Value =
        serde_json::from_str(&info_str_os).expect("Failed to parse OS info.json");
    let target_field_os = info_os["target"].as_str().expect("Target field missing in OS info.json").to_string();
    let rows_os = info_os["rows"].as_i64().expect("rows missing in OS info.json");
    let cols_os = info_os["columns"].as_i64().expect("columns missing in OS info.json");

    stdlib::types::set_size(rows_os, cols_os);
    let mut runtime_os = Runtime::new(1000, datadir_os.clone().into(), Some(device));
    runtime_os.dmgr.data_size = (rows_os as usize, cols_os as usize);
    let backtester_os = BasicBacktest::new(&mut runtime_os.dmgr, &target_field_os, metric);

    // 3. Resolve Output File Paths
    let path_is = args.output_file_is.unwrap_or_else(|| {
        if args.output_file.ends_with(".pnl") {
            args.output_file.replace(".pnl", "_is.pnl")
        } else if args.output_file.ends_with(".csv") {
            args.output_file.replace(".csv", "_is.csv")
        } else {
            format!("{}_is", args.output_file)
        }
    });

    let path_os = args.output_file_os.unwrap_or_else(|| {
        if args.output_file.ends_with(".pnl") {
            args.output_file.replace(".pnl", "_os.pnl")
        } else if args.output_file.ends_with(".csv") {
            args.output_file.replace(".csv", "_os.csv")
        } else {
            format!("{}_os", args.output_file)
        }
    });

    let path_is_tvr = path_is.replace(".pnl", ".tvr").replace(".csv", ".tvr");
    let path_os_tvr = path_os.replace(".pnl", ".tvr").replace(".csv", ".tvr");

    println!("In-Sample PnL output file: {:?}", path_is);
    println!("In-Sample Turnover output file: {:?}", path_is_tvr);
    println!("Out-Sample PnL output file: {:?}", path_os);
    println!("Out-Sample Turnover output file: {:?}", path_os_tvr);

    let mut out_file_is = File::create(&path_is)
        .unwrap_or_else(|_| panic!("Failed to create IS output file: {}", path_is));
    let mut out_file_os = File::create(&path_os)
        .unwrap_or_else(|_| panic!("Failed to create OS output file: {}", path_os));

    let mut out_tvr_is = File::create(&path_is_tvr)
        .unwrap_or_else(|_| panic!("Failed to create IS turnover file: {}", path_is_tvr));
    let mut out_tvr_os = File::create(&path_os_tvr)
        .unwrap_or_else(|_| panic!("Failed to create OS turnover file: {}", path_os_tvr));

    let write_legacy = args.output_file != path_is && args.output_file != path_os;
    let mut out_file_legacy = if write_legacy {
        Some(File::create(&args.output_file)
            .unwrap_or_else(|_| panic!("Failed to create legacy output file: {}", args.output_file)))
    } else {
        None
    };

    // 4. Evaluate Equations
    println!("--- Reading equations file: {:?} ---", args.equations_file);
    let file = File::open(&args.equations_file)
        .unwrap_or_else(|_| panic!("Failed to open equations file: {}", args.equations_file));
    let reader = BufReader::new(file);

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

        let expr_escaped = expr_str.replace("\"", "\"\"");

        // Evaluate In-Sample
        stdlib::types::set_size(rows_is, cols_is);
        let signal_is = runtime_is.lookup_or_run(&eq_network, eq_network.root);
        if let Signal::DataFrame(Some(tensor)) = signal_is {
            let returns_tensor = backtester_is.calc_returns(tensor);
            let vals = Vec::<f64>::try_from(returns_tensor.contiguous().to_kind(tch::Kind::Double))
                .expect("Failed to convert IS returns tensor to Vec<f64>");
            let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            let csv_line = format!("\"{}\",{}\n", expr_escaped, vals_str.join(","));
            out_file_is
                .write_all(csv_line.as_bytes())
                .expect("Failed to write to IS output file");

            if let Some(ref mut f_leg) = out_file_legacy {
                f_leg
                    .write_all(csv_line.as_bytes())
                    .expect("Failed to write to legacy output file");
            }

            // Turnover IS
            let tvr_tensor = backtester_is.calc_turnover(tensor);
            let tvr_vals = Vec::<f64>::try_from(tvr_tensor.contiguous().to_kind(tch::Kind::Double))
                .expect("Failed to convert IS turnover tensor to Vec<f64>");
            let tvr_str: Vec<String> = tvr_vals.iter().map(|v| v.to_string()).collect();
            let tvr_line = format!("\"{}\",{}\n", expr_escaped, tvr_str.join(","));
            out_tvr_is
                .write_all(tvr_line.as_bytes())
                .expect("Failed to write to IS turnover file");
        } else {
            eprintln!("Expression '{}' did not evaluate to a DataFrame on IS", expr_str);
            continue;
        }

        // Evaluate Out-Sample
        stdlib::types::set_size(rows_os, cols_os);
        let signal_os = runtime_os.lookup_or_run(&eq_network, eq_network.root);
        if let Signal::DataFrame(Some(tensor)) = signal_os {
            let returns_tensor = backtester_os.calc_returns(tensor);
            let vals = Vec::<f64>::try_from(returns_tensor.contiguous().to_kind(tch::Kind::Double))
                .expect("Failed to convert OS returns tensor to Vec<f64>");
            let vals_str: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            let csv_line = format!("\"{}\",{}\n", expr_escaped, vals_str.join(","));
            out_file_os
                .write_all(csv_line.as_bytes())
                .expect("Failed to write to OS output file");

            // Turnover OS
            let tvr_tensor = backtester_os.calc_turnover(tensor);
            let tvr_vals = Vec::<f64>::try_from(tvr_tensor.contiguous().to_kind(tch::Kind::Double))
                .expect("Failed to convert OS turnover tensor to Vec<f64>");
            let tvr_str: Vec<String> = tvr_vals.iter().map(|v| v.to_string()).collect();
            let tvr_line = format!("\"{}\",{}\n", expr_escaped, tvr_str.join(","));
            out_tvr_os
                .write_all(tvr_line.as_bytes())
                .expect("Failed to write to OS turnover file");
        } else {
            eprintln!("Expression '{}' did not evaluate to a DataFrame on OS", expr_str);
            continue;
        }

        count += 1;
    }

    println!(
        "Successfully evaluated {} equations across In-Sample ({:?}) and Out-Sample ({:?}) with turnover (.tvr)",
        count, path_is, path_os
    );
}
