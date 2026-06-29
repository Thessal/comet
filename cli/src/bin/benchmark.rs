use clap::Parser;
use parser::ast::{Network, NodeType};
use parser::behavior::BehaviorDecl;
use rl::action::ActionSpace;
use rl::model::{AgentModel, Model};
use runtime::runtime::Runtime;
use std::fs;
use std::time::Instant;
use tch::Device;
use tch::Tensor;
use tch::nn::OptimizerConfig;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value = "../examples/behavior_2.cm")]
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

    println!("Using device: {:?}", device);

    // 1. Measure sample expression calculation
    measure_sample_expression(device);

    // 2. Measure RL inference and training
    measure_rl_bottleneck(&args.file, device);
}

fn measure_sample_expression(device: Device) {
    let mut runtime = Runtime::new(1000, "./wrds/data".into(), Some(device));

    let src = r#"
    Behavior Bench(d: DataFrame) {
        operators=[add, rank, ts_rank]
    } -> DataFrame
    Flow sample {
        add(ts_rank(data("close"), 10), rank(data("close")))
    }
    "#;
    let (network, _) = parser::parser::parse(src).unwrap();

    // Warmup
    runtime.lookup_or_run(&network, network.root);

    let iters = 10;
    let start = Instant::now();
    let mut rt = Runtime::new(1000, "./wrds/data".into(), Some(device));
    for _ in 0..iters {
        rt.lookup_or_run(&network, network.root);
        assert!(rt.expr_cache.contains("data(\"close\")"));
        assert!(rt.expr_cache.contains("ts_rank(data(\"close\"), 10)"));
        assert!(
            rt.expr_cache
                .contains("add(ts_rank(data(\"close\"), 10), rank(data(\"close\")))")
        );
        // for (k, v) in rt.expr_cache.iter() {
        //     println!("{}", k);
        // }
        // rt.expr_cache.pop("data(\"close\")");
        rt.expr_cache.pop("ts_rank(data(\"close\"), 10)");
        rt.expr_cache.pop("rank(data(\"close\"))");
        rt.expr_cache
            .pop("add(ts_rank(data(\"close\"), 10), rank(data(\"close\")))");
    }
    let elapsed = start.elapsed() / iters;
    println!("--- 1. Sample Expression Calculation ---");
    println!("Expression: add(ts_rank(data(\"close\"), 10), rank(data(\"close\")))");
    println!("Average Time: {:?}", elapsed);
    println!();
    
    // Benchmark just rank(data("close"))
    let src2 = r#"
    Behavior Bench(d: DataFrame) {
        operators=[rank]
    } -> DataFrame
    Flow sample {
        rank(data("close"))
    }
    "#;
    let (network2, _) = parser::parser::parse(src2).unwrap();
    rt.lookup_or_run(&network2, network2.root);
    let start2 = Instant::now();
    for _ in 0..iters {
        rt.lookup_or_run(&network2, network2.root);
        rt.expr_cache.pop("rank(data(\"close\"))");
    }
    let elapsed2 = start2.elapsed() / iters;
    println!("Expression: rank(data(\"close\"))");
    println!("Average Time: {:?}", elapsed2);
    println!();

    // Benchmark just ts_rank(data("close"), 10)
    let src3 = r#"
    Behavior Bench(d: DataFrame) {
        operators=[ts_rank]
    } -> DataFrame
    Flow sample {
        ts_rank(data("close"), 10)
    }
    "#;
    let (network3, _) = parser::parser::parse(src3).unwrap();
    rt.lookup_or_run(&network3, network3.root);
    let start3 = Instant::now();
    for _ in 0..iters {
        rt.lookup_or_run(&network3, network3.root);
        rt.expr_cache.pop("ts_rank(data(\"close\"), 10)");
    }
    let elapsed3 = start3.elapsed() / iters;
    println!("Expression: ts_rank(data(\"close\"), 10)");
    println!("Average Time: {:?}", elapsed3);
    println!();
    
    // Benchmark just ts_corr(data("close"), data("open"), 10)
    let src4 = r#"
    Behavior Bench(d: DataFrame) {
        operators=[ts_corr]
    } -> DataFrame
    Flow sample {
        ts_corr(data("close"), data("open"), 10)
    }
    "#;
    let (network4, _) = parser::parser::parse(src4).unwrap();
    rt.lookup_or_run(&network4, network4.root);
    let start4 = Instant::now();
    for _ in 0..iters {
        rt.lookup_or_run(&network4, network4.root);
        rt.expr_cache.pop("ts_corr(data(\"close\"), data(\"open\"), 10)");
    }
    let elapsed4 = start4.elapsed() / iters;
    println!("Expression: ts_corr(data(\"close\"), data(\"open\"), 10)");
    println!("Average Time: {:?}", elapsed4);
    println!();
    
    // Benchmark just ts_stddev(data("close"), 10)
    let src5 = r#"
    Behavior Bench(d: DataFrame) {
        operators=[ts_stddev]
    } -> DataFrame
    Flow sample {
        ts_stddev(data("close"), 10)
    }
    "#;
    let (network5, _) = parser::parser::parse(src5).unwrap();
    rt.lookup_or_run(&network5, network5.root);
    let start5 = Instant::now();
    for _ in 0..iters {
        rt.lookup_or_run(&network5, network5.root);
        rt.expr_cache.pop("ts_stddev(data(\"close\"), 10)");
    }
    let elapsed5 = start5.elapsed() / iters;
    println!("Expression: ts_stddev(data(\"close\"), 10)");
    println!("Average Time: {:?}", elapsed5);
    println!();
}

fn measure_rl_bottleneck(filename: &str, device: Device) {
    println!("--- 2. RL Inference & Training Bottleneck ---");
    let src = fs::read_to_string(filename).expect("Failed to read file");
    let (network, behavior_nodes) = parser::parser::parse(&src).expect("Parse failed");

    let behavior_decl: &BehaviorDecl = match &network.nodes[behavior_nodes[0]].node_type {
        NodeType::Behavior(b) => b,
        _ => unreachable!(),
    };
    let action_space: ActionSpace = behavior_decl.into();

    let mut runtime = Runtime::new(300, "./wrds/data".into(), Some(device));
    let seq_len = 20;

    let vs = tch::nn::VarStore::new(device);
    let mut model = AgentModel::new(&vs.root(), action_space.clone(), 256);
    let mut opt = tch::nn::Adam::default().build(&vs, 1e-4).unwrap();

    let mut env = rl::env::Environment::new(
        &network,
        action_space.clone(),
        rl::pool::Pool::new(
            runtime::backtest::BasicBacktest::new(&mut runtime.dmgr, "returns_d1"),
            device,
            1.0,
        ),
        seq_len,
        1,
    );

    env.reset();
    let mask = env.get_valid_action_mask(&device);
    let actions = vec![0, 1, 2]; // dummy actions

    let _ = tch::no_grad(|| model.forward(&env.state, &mut runtime, &mask, &device, &actions));

    let iters = 10;
    let mut total_embed = std::time::Duration::new(0, 0);
    let mut total_decoder = std::time::Duration::new(0, 0);

    for _ in 0..iters {
        let (stack, callgraph): (&Vec<(stdlib::types::Signal, usize)>, &Network) =
            env.state.machine.get_stack();
        let mut data_tensors = Vec::new();

        let t_embed = Instant::now();
        for (_signal_decl, addr) in stack.iter() {
            let signal = runtime.lookup_or_run(callgraph, *addr);
            if let stdlib::types::Signal::DataFrame(Some(df)) = signal {
                data_tensors.push(df.to_device(tch::Device::Cpu));
            }
        }

        let mut alpha_matrix = if data_tensors.is_empty() {
            Tensor::zeros(
                [stdlib::types::SIZE[0], stdlib::types::SIZE[1]],
                (tch::Kind::Float, device),
            )
        } else {
            let stacked = Tensor::stack(&data_tensors, 0);
            let size = stacked.size();
            let flattened: Vec<f32> = stacked.flatten(0, -1).try_into().unwrap_or_default();
            let stacked = Tensor::from_slice(&flattened)
                .view([size[0], size[1], size[2]])
                .to(device);
            stacked.mean_dim(Some([0].as_slice()), false, tch::Kind::Float)
        };
        alpha_matrix = alpha_matrix.nan_to_num(0.0, 0.0, 0.0);
        let alpha_matrix = alpha_matrix.unsqueeze(0);

        let mut target_tokens = actions.to_vec();
        target_tokens.insert(0, 0);
        target_tokens.truncate(50);
        while target_tokens.len() < 50 {
            target_tokens.push(0);
        }
        let shifted_target_tokens = Tensor::from_slice(&target_tokens).to(device).unsqueeze(0);
        total_embed += t_embed.elapsed();

        let t_decode = Instant::now();
        let _ = model.decoder.forward(&shifted_target_tokens, &alpha_matrix);
        total_decoder += t_decode.elapsed();
    }

    println!("Inference - DataFrame Embedding: {:?}", total_embed / iters);
    println!(
        "Inference - Transformer Decoder: {:?}",
        total_decoder / iters
    );

    // Measure Training (Backward Pass)
    let iters_train = 10;
    let mut total_backward = std::time::Duration::new(0, 0);
    for _ in 0..iters_train {
        let logits = model.decoder.forward(
            &Tensor::zeros([32, 50], (tch::Kind::Int64, device)),
            &Tensor::zeros(
                [32, stdlib::types::SIZE[0], stdlib::types::SIZE[1]],
                (tch::Kind::Float, device),
            ),
        );
        let loss = logits.mean(tch::Kind::Float);

        let t_back = Instant::now();
        opt.backward_step_clip(&loss, 0.5);
        total_backward += t_back.elapsed();
    }

    println!(
        "Training - Backward pass (batch=32): {:?}",
        total_backward / iters_train
    );
}
