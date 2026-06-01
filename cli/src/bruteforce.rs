use rl::env::Environment;

pub struct BruteforceSearch {
    env: Environment,
}

// impl Search for BruteforceSearch {
//     fn new(mut runtime: Runtime, ast: Tree, behavior_nodes: Vec<BehaviorNode>) -> Self {
//         let score_fn = runtime::stats::Aggregator::new();

//         Self {
//             env: Environment::new(&mut runtime, ast, behavior_nodes, score_fn, 64, 1024),
//             result: SearchResult::new(),
//         }
//     }

//     fn search(&self) -> Vec<Sequence> {
//         todo!()
//     }
// }

// fn brute_force(
//     network: Network,
//     root: usize,
//     behavior_decls: Vec<BehaviorDecl>,
//     behavior_nodes: Vec<usize>,
// ) {
//     use rand::seq::SliceRandom;
//     use rand::thread_rng;

//     let mut runtime = runtime::runtime::Runtime::new(10000, "data".into(), None);
//     runtime.enable = false;

//     let score_fn = runtime::stats::Aggregator {
//         weights: std::collections::HashMap::new(),
//     };

//     let behavior_decl = behavior_decls
//         .first()
//         .expect("No behavior decl found")
//         .clone();
//     let behavior_node_id = *behavior_nodes.first().expect("No behavior node found");
//     let params = network.nodes[behavior_node_id].children.clone();

//     let mut env = rl::env::Environment::new(&mut runtime, behavior_decl, params, score_fn, 20, 1);

//     let mut results = Vec::new();
//     let mut rng = thread_rng();

//     while results.len() < 3 {
//         env.reset();
//         let mut finished = false;
//         let mut steps = 0;
//         let mut sequence = Vec::new();

//         while !finished && steps < 20 {
//             let valid_actions = env.state.get_valid_actions(&env.action_space);
//             if valid_actions.is_empty() {
//                 break;
//             }

//             let action = valid_actions.choose(&mut rng).unwrap().clone();
//             if let rl::action::Action::Done = action {
//                 finished = true;
//             }

//             let step = env.step(action);
//             sequence = step.sequence.clone();
//             steps += 1;
//         }

//         if finished && !results.contains(&sequence) {
//             println!("Found valid sequence {}: {:?}", results.len() + 1, sequence);
//             results.push(sequence);
//         }
//     }
// }
