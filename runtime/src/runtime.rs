use crate::ast::{Program, Token, Tree}; // Vec<Token>
use crate::dmgr::DataManager;
use lru::LruCache;
use parser::program::Literal;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use stdlib::Signal;

pub struct Runtime {
    pub dmgr: DataManager,
    pub expr_cache: LruCache<String, Signal>,
    pub expr_lookups: usize,
    pub expr_hits: usize,
}

use crate::ast::OperatorSpec;

impl Runtime {
    pub fn new(capacity: usize, data_dir: PathBuf) -> Self {
        Runtime {
            dmgr: DataManager::new(data_dir),
            expr_cache: LruCache::new(NonZeroUsize::new(capacity).unwrap()), // TODO : evict only when memory is full
            expr_lookups: 0,
            expr_hits: 0,
        }
    }

    fn load_data(&mut self, program: &Program) -> Result<&Signal, String> {
        // Loads data in to cache
        let key: String = program.into();
        let first_arg: &Tree = program
            .parameters
            .as_ref()
            .unwrap()
            .into_iter()
            .nth(0)
            .unwrap();
        let data_name: String = match first_arg {
            Tree::Literal(Literal::String(literal)) => literal.clone(),
            _ => panic!("First argument of data operator is not a string"),
        };
        let data = || -> Result<Signal, String> {
            Ok(Signal::DataFrame(Some(
                self.dmgr.get_data(data_name.as_str()),
            )))
        };
        self.expr_cache.try_get_or_insert(key, data)
    }

    pub fn run(&mut self, ast: &Tree) -> Signal {
        match ast {
            Tree::Program(program) => {
                let polish_expr: String = program.into();
                if program.spec.name == "data" {
                    self.load_data(program).unwrap();
                }
                match self.expr_cache.get(&polish_expr) {
                    Some(signal) => signal.clone(),
                    None => {
                        let args: Vec<Signal> = program
                            .parameters
                            .clone()
                            .unwrap()
                            .into_iter()
                            .map(|param| self.run(&param))
                            .collect();
                        let result = self.evaluate(program.spec.clone(), args).unwrap();
                        self.expr_cache.put(polish_expr, result.clone());
                        result
                    }
                }
            }
            Tree::Literal(Literal::Boolean(literal)) => panic!("Boolean literal not supported"),
            Tree::Literal(Literal::Integer(literal)) => Signal::Int(Some(literal.clone())),
            Tree::Literal(Literal::Float(literal)) => Signal::Float(Some(literal.clone())),
            Tree::Literal(Literal::String(literal)) => Signal::String(Some(literal.clone())),
        }
    }

    pub fn evaluate(&self, spec: OperatorSpec, args: Vec<Signal>) -> Result<Signal, String> {
        let operator: stdlib::OperatorMeta = (spec.name.as_str()).into();
        operator.execute(&args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::PolishExpr;

    #[test]
    fn test_runtime_minimal() {
        // volume / ts_mean(volume, 10)
        // [ "volume", "data", "10", "volume", "data", "ts_mean", "divide" ]
        let expr: PolishExpr = vec![
            Token::Literal(Literal::String("volume".to_string())),
            Token::Operator("data".into()),
            Token::Literal(Literal::Integer(10)),
            Token::Literal(Literal::String("volume".to_string())),
            Token::Operator("data".into()),
            Token::Operator("ts_mean".into()),
            Token::Operator("divide".into()),
        ];
        let program: Tree = (&expr).into();

        match program.clone() {
            Tree::Program(program) => {
                let tokens: Vec<String> = program
                    .polish_expression
                    .unwrap()
                    .iter()
                    .map(|x| x.into())
                    .collect();
                // println!("{:?}", tokens);
                assert!(
                    tokens
                        == [
                            "str!volume",
                            "op!data",
                            "int!10",
                            "str!volume",
                            "op!data",
                            "op!ts_mean",
                            "op!divide"
                        ]
                )
            }
            _ => panic!("Program is not a program"),
        }

        let mut runtime = Runtime::new(100, "../data".into());

        let result = runtime.run(&program);
        match result {
            Signal::DataFrame(Some(df)) => {
                assert!(
                    df.len() > 0,
                    "Execution should yield a non-empty result DataFrame array"
                );
                // println!("Execution Result Size: {} x {}", df.len(), df[0].len());
            }
            _ => {
                todo!("Execution result is not a DataFrame");
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_nonexisting_data() {
        // data("nonexistingdata")
        // [ "nonexistingdata", "data"]
        let expr: PolishExpr = vec![
            Token::Literal(Literal::String("nonexistingdata".to_string())),
            Token::Operator("data".into()),
        ];
        let program: Tree = (&expr).into();

        let mut runtime = Runtime::new(100, "../data".into());

        let _result = runtime.run(&program);
    }
}
