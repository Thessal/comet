pub trait Search {
    pub fn new(runtime: Runtime, ast: Tree, behavior_nodes: Vec<BehaviorNode>) -> Self;
    pub fn search(&self) -> Vec<Sequence>;
}

pub struct SearchResult {
    pub trees: Vec<Tree>,
    pub rpns: Vec<Vec<Token>>,
    // pub actions:
    pub score: Option<Vec<f64>>,
}

impl SearchResult {
    pub fn new() -> Self {
        Self {
            trees: Vec::new(),
            rpns: Vec::new(),
            score: None,
        }
    }
}
