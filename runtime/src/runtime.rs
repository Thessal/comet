use crate::dmgr::DataManager;
use stdlib::ParamType;

pub struct Runtime {
    pub dmgr: DataManager,
}

// #[derive(Debug, Clone, PartialEq)]
// pub enum ParamType {
//     Numeric(f64),
//     String(String),
//     Data(String),
// }

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
    ) -> Result<ParamType, String> {
        let mut stack: Vec<ParamType> = Vec::new();

        for token in seq {
            if token == "!shift" {
                if let Some(param_name) = param_names.pop() {
                    let first_char = param_name.chars().next().unwrap_or('\0');
                    if first_char.is_ascii_digit() || first_char == '-' || first_char == '.' {
                        let number = param_name.parse::<f64>().map_err(|e| e.to_string())?;
                        stack.push(ParamType::Float(number));
                    } else if first_char == '"' || first_char == '\'' {
                        stack.push(ParamType::String(
                            param_name[1..param_name.len() - 1].to_string(),
                        ));
                    } else if first_char.is_ascii_alphabetic() {
                        stack.push(ParamType::Variable(param_name));
                    } else {
                        return Err(format!(
                            "Unrecognized boundary format for param: {}",
                            param_name
                        ));
                    }
                } else {
                    return Err("Shift without available parameters".into());
                }
            } else if token == "data" {
                if let Some(ParamType::String(name)) = stack.pop() {
                    let data = self.dmgr.get_data(&name);
                    if data.len() == 1 {
                        stack.push(ParamType::Vector(data.into_iter().next().unwrap()));
                    } else {
                        stack.push(ParamType::DataFrame(data));
                    }
                } else {
                    return Err("Data operator expects String on stack".into());
                }
            } else {
                let first_char = token.chars().next().unwrap_or('\0');
                if first_char.is_ascii_digit() || first_char == '-' || first_char == '.' {
                    let f = token.parse::<f64>().map_err(|e| e.to_string())?;
                    stack.push(ParamType::Float(f));
                } else if first_char == '"' || first_char == '\'' {
                    return Err(format!("String sequence tokens unsupported: {}", token));
                } else if first_char.is_ascii_alphabetic() {
                    let func_name = token;

                    let mut out = ParamType::Vector(vec![]);
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
                                let mut arg = stack.pop().unwrap();
                                if let ParamType::Variable(name) = &arg {
                                    let data = self.dmgr.get_data(name);
                                    if data.len() == 1 {
                                        arg = ParamType::Vector(data.into_iter().next().unwrap());
                                    } else {
                                        arg = ParamType::DataFrame(data);
                                    }
                                }
                                args.push(arg);
                            }
                            args.reverse();

                            let mut type_mismatch = false;
                            for (i, arg) in args.iter().enumerate() {
                                let expected = &meta.inputs[i];
                                let ok = match expected {
                                    stdlib::OutputShape::Void => false,
                                    stdlib::OutputShape::TimeSeries | stdlib::OutputShape::Vector => {
                                        matches!(arg, ParamType::Vector(_))
                                    }
                                    stdlib::OutputShape::DataFrame => {
                                        matches!(arg, ParamType::DataFrame(_))
                                    }
                                    stdlib::OutputShape::Matrix => false,
                                    stdlib::OutputShape::ScalarFloat | stdlib::OutputShape::ScalarInt => {
                                        matches!(arg, ParamType::Float(_))
                                    }
                                    stdlib::OutputShape::ScalarString => {
                                        matches!(arg, ParamType::String(_))
                                    }
                                };
                                if !ok {
                                    type_mismatch = true;
                                    break;
                                }
                            }
                            if type_mismatch {
                                return Err(format!("Type mismatch for {}", func_name));
                            }

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
                } else {
                    return Err(format!("Unrecognized sequence token: {}", token));
                }
            }
        }

        if stack.len() != 1 {
            return Err(format!(
                "Final stack size is not 1 (size = {})",
                stack.len()
            ));
        }

        stack.pop().ok_or("Stack is empty".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_1_execution() {
        let mut runtime = Runtime::new(100, "../data");

        // volume / ts_mean(volume, 10)
        // Native stack execution: evaluate "volume", "data", "10", "volume", "data", "ts_mean", "divide" natively!
        // To do this, the sequence follows Reverse Polish standard:
        // But since param names are popped from end of `param_names`, the original system supplied param names reversed!

        let param_names = vec![
            "\"volume\"".to_string(),
            "10".to_string(),
            "\"volume\"".to_string(),
        ];

        // FIXME : if input is the following, it is divide(data(volume), ts_mean(10, data(volume))) , which is malformed and must generate error.
        // let seq_bad = vec![
        //     "!shift".to_string(),
        //     "data".to_string(),
        //     "!shift".to_string(),
        //     "ts_mean".to_string(),
        //     "!shift".to_string(),
        //     "data".to_string(),
        //     "divide".to_string(),
        // ];
        let seq = vec![
            "!shift".to_string(),
            "data".to_string(),
            "!shift".to_string(),
            "!shift".to_string(),
            "data".to_string(),
            "ts_mean".to_string(),
            "divide".to_string(),
        ];
        let result = runtime
            .evaluate_sequence(&seq, param_names)
            .expect("Evaluation succeeded");
        match result {
            ParamType::DataFrame(df) => {
                println!("Execution Result Size: {} x {}", df.len(), df[0].len());
                assert!(
                    df.len() > 0,
                    "Execution should yield a non-empty result DataFrame array"
                );
                for line in &df[0..20] {
                    println!("{:?}", line);
                }
            }
            _ => {
                todo!("Execution result is not a DataFrame");
            }
        }
    }

    #[test]
    fn test_malformed_sequence_type_mismatch() {
        let mut runtime = Runtime::new(100, "../data");

        let param_names = vec![
            "\"volume\"".to_string(),
            "10".to_string(),
            "\"volume\"".to_string(),
        ];

        let seq_bad = vec![
            "!shift".to_string(),
            "data".to_string(),
            "!shift".to_string(),
            "ts_mean".to_string(),
            "!shift".to_string(),
            "data".to_string(),
            "divide".to_string(),
        ];

        let result = runtime.evaluate_sequence(&seq_bad, param_names);
        assert!(result.is_err(), "Malformed sequence must generate an error");
    }
}
