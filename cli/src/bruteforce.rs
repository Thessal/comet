use rl::env::Environment;

pub struct BruteforceSearch {
    env: Environment,
    result: SearchResult,
}

impl Search for BruteforceSearch {
    fn new(mut runtime: Runtime, ast: Tree, behavior_nodes: Vec<BehaviorNode>) -> Self {
        let score_fn = runtime::stats::Aggregator::new();

        Self {
            env: Environment::new(&mut runtime, ast, behavior_nodes, score_fn, 64, 1024),
            result: SearchResult::new(),
        }
    }

    fn search(&self) -> Vec<Sequence> {
        todo!()
    }
}
