use crate::dmgr::DataManager;
use lru::LruCache;
use parser::ast::{Network, Node, NodeType};
use parser::expr::Literal;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use stdlib::types::Signal;

pub static mut RUNTIME: Option<Runtime> = None;

pub struct Runtime {
    pub dmgr: DataManager,
    pub expr_cache: LruCache<String, Signal>,
    pub expr_lookups: usize,
    pub expr_hits: usize,
    pub enable: bool,
}

use stdlib::OperatorSpec;

impl Runtime {
    pub fn new(capacity: usize, data_dir: PathBuf, device: Option<tch::Device>) -> Self {
        Runtime {
            dmgr: DataManager::new(data_dir, device),
            expr_cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()), // TODO : evict only when memory is full
            expr_lookups: 0,
            expr_hits: 0,
            enable: true,
        }
    }

    pub fn lookup_or_run(&mut self, callgraph: &Network, root: usize) -> &Signal {
        let hash_key: String = callgraph.format_node(root);

        //TODO: mask outside universe to nan. see universe.csv.gz
        if self.expr_cache.get(&hash_key).is_none() {
            let data = self.run(callgraph, root);
            self.expr_cache.put(hash_key.clone(), data);
        }
        self.expr_cache.get(&hash_key).unwrap()
    }

    fn data_operator(&mut self, network: &Network, node: &Node) -> Signal {
        let child_sig = self.run(network, node.children[0]);
        if let Signal::String(Some(s)) = child_sig {
            return Signal::DataFrame(Some(
                self.dmgr
                    .get_data(&s)
                    .expect("Data not found")
                    .shallow_clone(),
            ));
        } else {
            panic!("data operator requires a String argument");
        }
    }

    fn run(&mut self, network: &Network, root: usize) -> Signal {
        let node = &network.nodes[root];
        match &node.node_type {
            NodeType::Operator(spec) => {
                if spec.name == "data" {
                    self.data_operator(network, node)
                } else {
                    let args: Vec<Signal> = node
                        .children
                        .iter()
                        .map(|&child| self.run(network, child))
                        .collect();
                    self.execute(spec, args).unwrap()
                }
            }
            NodeType::Behavior(_) => panic!("Behavior node cannot be run"),
            NodeType::Literal(Literal::Boolean(_literal)) => {
                panic!("Boolean literal not supported")
            }
            NodeType::Literal(Literal::Integer(literal)) => Signal::Int(Some(literal.clone())),
            NodeType::Literal(Literal::Float(literal)) => Signal::Float(Some(literal.clone())),
            NodeType::Literal(Literal::String(literal)) => Signal::String(Some(literal.clone())),
        }
    }

    fn execute(&self, spec: &OperatorSpec, args: Vec<Signal>) -> Result<Signal, String> {
        spec.execute(&args)
    }
}

// pub fn test_make_param0() -> Program {
//     Program::new(
//         "data",
//         vec![Tree::Literal(Literal::String("volume".to_string()))],
//     )
// }
// pub fn test_make_param1() -> Program {
//     Program::new(
//         "data",
//         vec![Tree::Literal(Literal::String("close".to_string()))],
//     )
// }

// #[cfg(test)]
// pub mod tests {
//     use super::*;
//     use crate::ast::PolishExpr;

//     #[test]
//     fn test_runtime_minimal() {
//         let mut runtime = Runtime::new(100, "../data".into(), None);
//         let param0 = test_make_param0();
//         let params: Vec<Program> = vec![param0];

//         // volume / ts_mean(param0, 10)
//         // [ param0, "10", param0, "ts_mean", "divide" ]
//         let expr: PolishExpr = vec![
//             Token::Parameter(0),
//             Token::Literal(Literal::Integer(10)),
//             Token::Parameter(0),
//             Token::Operator("ts_mean".into()),
//             Token::Operator("divide".into()),
//         ];
//         let program: Tree = (&expr, params).into();

//         match program.clone() {
//             Tree::Program(program) => {
//                 let tokens: Vec<String> = program
//                     .polish_expression
//                     .unwrap()
//                     .iter()
//                     .map(|x| x.into())
//                     .collect();
//                 println!("{:?}", tokens);

//                 assert!(
//                     tokens
//                         == [
//                             "str!volume",
//                             "op!data",
//                             "int!10",
//                             "str!volume",
//                             "op!data",
//                             "op!ts_mean",
//                             "op!divide"
//                         ]
//                 )
//             }
//             _ => panic!("Program is not a program"),
//         }

//         let result = runtime.run(&program);
//         match result {
//             Signal::DataFrame(Some(df)) => {
//                 assert!(
//                     df.size2().unwrap().0 > 0 && df.size2().unwrap().1 > 0,
//                     "Execution should yield a non-empty result DataFrame array"
//                 );
//             }
//             _ => {
//                 todo!("Execution result is not a DataFrame");
//             }
//         }
//     }

//     #[test]
//     #[should_panic]
//     fn test_nonexisting_data() {
//         // data("nonexistingdata")
//         // [ "nonexistingdata", "data"]
//         let expr: PolishExpr = vec![
//             Token::Literal(Literal::String("nonexistingdata".to_string())),
//             Token::Operator("data".into()),
//         ];
//         let program: Tree = (&expr, vec![]).into();

//         let mut runtime = Runtime::new(100, "../data".into(), None);

//         let _result = runtime.run(&program);
//     }
// }
