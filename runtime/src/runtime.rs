use crate::dmgr::DataManager;

pub struct Runtime {
    pub dmgr: DataManager,
}

impl Runtime {
    pub fn new<P: AsRef<std::path::Path>>(_capacity: usize, data_dir: P) -> Self {
        Runtime {
            dmgr: DataManager::new(data_dir),
        }
    }

    pub fn evaluate_sequence(
        &mut self,
        seq: &[String],
        mut param_names: Vec<String>,
    ) -> Result<Vec<f64>, String> {
        let mut stack: Vec<Vec<f64>> = Vec::new();

        for token in seq {
            if token == "shift" {
                if let Some(param_name) = param_names.pop() {
                    let data = self.dmgr.get_data(&param_name);
                    stack.push(data);
                } else {
                    return Err("Shift without available parameters".into());
                }
            } else if let Ok(f) = token.parse::<f64>() {
                // If the token is a raw number (e.g. integer parameters)
                stack.push(vec![f; 1]);
            } else {
                let func_name = token;
                
                let mut out = vec![];
                let mut is_void = false;
                let mut found = false;

                for meta in inventory::iter::<stdlib::OperatorMeta> {
                    if meta.name == func_name.as_str() {
                        let arity = meta.inputs.len();
                        if stack.len() < arity {
                            return Err(format!("Stack underflow for {}", func_name));
                        }

                        let mut args = Vec::with_capacity(arity);
                        for _ in 0..arity {
                            args.push(stack.pop().unwrap());
                        }
                        args.reverse();

                        out = (meta.execute)(&args);
                        is_void = meta.output_shape == stdlib::OutputShape::Void;
                        found = true;
                        break;
                    }
                }

                if !found {
                    return Err(format!("Unknown function in sequence: {}", func_name));
                }

                if !is_void {
                    stack.push(out);
                }
            }
        }

        if stack.len() != 1 {
            return Err(format!("Final stack size is not 1 (size = {})", stack.len()));
        }

        Ok(stack.pop().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_1_execution() {
        let mut runtime = Runtime::new(100, "../data");

        // Native stack execution: evaluate "volume", "10", "ts_mean", "volume", "divide" natively!
        // To do this, the sequence follows Reverse Polish standard:
        // [shift(volume), shift(10), ts_mean, shift(volume), divide]
        // But since param names are popped from end of `param_names`, the original system supplied param names reversed!
        
        let available_funcs = codegen::codegen::get_available_functions();
        let param_names = vec!["volume".to_string(), "10".to_string(), "volume".to_string()];
        
        let seq = vec![
            "shift".to_string(),
            "shift".to_string(),
            "ts_mean".to_string(),
            "shift".to_string(),
            "divide".to_string(),
        ];

        let result = runtime.evaluate_sequence(&seq, param_names, &available_funcs).expect("Evaluation succeeded");
        println!("Execution Result Length: {}", result.len());
        assert!(result.len() > 0, "Execution should yield a non-empty result DataFrame array");
    }
}
