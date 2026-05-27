use crate::ast::{Program, Token, Tree};
use crate::dmgr::DataManager;
use lru::LruCache;
use parser::expr::Literal;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use stdlib::types::Signal;

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
            Ok(Signal::DataFrame(self.dmgr.get_data(data_name.as_str())))
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
                        let args: Vec<Signal> = match &program.parameters {
                            Some(x) => x.iter().map(|param| self.run(param)).collect(),
                            None => vec![],
                        };
                        let result = self.evaluate(program.spec.clone(), args).unwrap();
                        self.expr_cache.put(polish_expr, result.clone());
                        result
                    }
                }
            }
            Tree::Literal(Literal::Boolean(_literal)) => panic!("Boolean literal not supported"),
            Tree::Literal(Literal::Integer(literal)) => Signal::Int(Some(literal.clone())),
            Tree::Literal(Literal::Float(literal)) => Signal::Float(Some(literal.clone())),
            Tree::Literal(Literal::String(literal)) => Signal::String(Some(literal.clone())),
        }
    }

    fn evaluate(&self, spec: OperatorSpec, args: Vec<Signal>) -> Result<Signal, String> {
        if !self.enable {
            Ok(Signal::DataFrame(Some(tch::Tensor::zeros(
                &[1, 1],
                tch::kind::FLOAT_CPU,
            ))))
        } else {
            spec.execute(&args)
        }
    }
}

pub fn test_make_param0() -> Program {
    Program::new(
        "data",
        vec![Tree::Literal(Literal::String("volume".to_string()))],
    )
}
pub fn test_make_param1() -> Program {
    Program::new(
        "data",
        vec![Tree::Literal(Literal::String("close".to_string()))],
    )
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::ast::PolishExpr;

    #[test]
    fn test_runtime_minimal() {
        let mut runtime = Runtime::new(100, "../data".into(), None);
        let param0 = test_make_param0();
        let params: Vec<Program> = vec![param0];

        // volume / ts_mean(param0, 10)
        // [ param0, "10", param0, "ts_mean", "divide" ]
        let expr: PolishExpr = vec![
            Token::Parameter(0),
            Token::Literal(Literal::Integer(10)),
            Token::Parameter(0),
            Token::Operator("ts_mean".into()),
            Token::Operator("divide".into()),
        ];
        let program: Tree = (&expr, params).into();

        match program.clone() {
            Tree::Program(program) => {
                let tokens: Vec<String> = program
                    .polish_expression
                    .unwrap()
                    .iter()
                    .map(|x| x.into())
                    .collect();
                println!("{:?}", tokens);

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

        let result = runtime.run(&program);
        match result {
            Signal::DataFrame(Some(df)) => {
                assert!(
                    df.size2().unwrap().0 > 0 && df.size2().unwrap().1 > 0,
                    "Execution should yield a non-empty result DataFrame array"
                );
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
        let program: Tree = (&expr, vec![]).into();

        let mut runtime = Runtime::new(100, "../data".into(), None);

        let _result = runtime.run(&program);
    }
}
