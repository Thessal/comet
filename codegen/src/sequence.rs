use crate::dag::{DagBuilder, DagOp};
use parser::program::Literal;

/// Converts a DAG expression into a Polish notation sequence.
/// `root_id` is the starting node in the DagBuilder.
pub fn dag_to_sequence(dag: &DagBuilder, root_id: usize) -> Vec<String> {
    let mut seq = Vec::new();
    let node = &dag.nodes[root_id];
    match &node.op {
        DagOp::Literal(lit) => {
            seq.push(match lit {
                Literal::Integer(i) => i.to_string(),
                Literal::Float(f) => f.to_string(),
                Literal::String(s) => format!("\"{}\"", s), // Quote strings to distinguish
                Literal::Boolean(b) => b.to_string(),
            });
        }
        DagOp::Identifier(ident) => {
            seq.push(ident.clone());
        }
        DagOp::CallFn { func_name, args } => {
            // Include arity in the sequence for easier reconstruction
            // Format: CALL|func_name|arity
            seq.push(format!("CALL|{}|{}", func_name, args.len()));
            for child_id in args {
                seq.push("ARG|".to_string());
                let mut child_seq = dag_to_sequence(dag, *child_id);
                seq.append(&mut child_seq);
            }
        }
    }
    seq
}

/// Reconstructs a DAG from a Polish notation sequence.
/// Returns the DagBuilder and the root node ID.
pub fn sequence_to_dag(seq: &[String]) -> Result<(DagBuilder, usize), String> {
    let mut builder = DagBuilder::new();
    let mut pos = 0;
    if seq.is_empty() {
        return Err("Empty sequence".to_string());
    }
    let root_id = parse_sequence_impl(seq, &mut pos, &mut builder)?;
    if pos != seq.len() {
        return Err("Sequence not fully consumed".to_string());
    }
    Ok((builder, root_id))
}

pub fn parse_sequence_impl(
    seq: &[String],
    pos: &mut usize,
    builder: &mut DagBuilder,
) -> Result<usize, String> {
    if *pos >= seq.len() {
        return Err("Unexpected end of sequence".to_string());
    }
    let token = &seq[*pos];
    *pos += 1;

    if token.starts_with("CALL|") {
        let parts: Vec<&str> = token.split('|').collect();
        if parts.len() != 3 {
            return Err(format!("Invalid CALL token: {}", token));
        }
        let func_name = parts[1].to_string();
        let arity: usize = parts[2].parse().map_err(|_| "Invalid arity".to_string())?;

        let mut args = Vec::new();
        for _ in 0..arity {
            if *pos >= seq.len() {
                return Err("Unexpected end of sequence reading args".to_string());
            }
            let arg_token = &seq[*pos];
            *pos += 1;
            if !arg_token.starts_with("ARG|") {
                return Err(format!("Expected ARG token, got: {}", arg_token));
            }

            let child_id = parse_sequence_impl(seq, pos, builder)?;
            args.push(child_id);
        }
        Ok(builder.insert_op(DagOp::CallFn { func_name, args }))
    } else if token.starts_with('"') && token.ends_with('"') {
        let s = token[1..token.len() - 1].to_string();
        Ok(builder.insert_op(DagOp::Literal(Literal::String(s))))
    } else if let Ok(i) = token.parse::<i64>() {
        Ok(builder.insert_op(DagOp::Literal(Literal::Integer(i))))
    } else if let Ok(f) = token.parse::<f64>() {
        Ok(builder.insert_op(DagOp::Literal(Literal::Float(f))))
    } else if token == "true" {
        Ok(builder.insert_op(DagOp::Literal(Literal::Boolean(true))))
    } else if token == "false" {
        Ok(builder.insert_op(DagOp::Literal(Literal::Boolean(false))))
    } else {
        Ok(builder.insert_op(DagOp::Identifier(token.clone())))
    }
}

use parser::program::TypeDecl;

// ... parsing old prefix standard ...

pub fn action_sequence_to_dag(
    seq: &[String],
    mut param_names: Vec<String>, // Unprocessed parameters (reversed!)
    available_funcs: &[(String, Vec<TypeDecl>, TypeDecl)],
    builder: &mut DagBuilder,
) -> Result<usize, String> {
    let mut stack = Vec::new();

    for token in seq {
        if token == "shift" {
            if let Some(param_name) = param_names.pop() {
                let node_id = builder.insert_op(DagOp::Identifier(param_name));
                stack.push(node_id);
            } else {
                return Err("Shift without available parameters".into());
            }
        } else {
            let func_name = token;
            let (arity, ret_type) = match available_funcs.iter().find(|f| &f.0 == func_name) {
                Some(f) => (f.1.len(), f.2.clone()),
                None => return Err(format!("Unknown function in sequence: {}", func_name)),
            };

            if stack.len() < arity {
                return Err("Stack underflow during reduce".into());
            }

            let mut args = Vec::new();
            for _ in 0..arity {
                args.push(stack.pop().unwrap());
            }
            args.reverse();

            let node_id = builder.insert_op(DagOp::CallFn {
                func_name: func_name.clone(),
                args,
            });

            if ret_type != TypeDecl::Void {
                stack.push(node_id);
            }
        }
    }

    if stack.len() != 1 {
        return Err(format!(
            "Final stack size is not 1 (size = {})",
            stack.len()
        ));
    }

    Ok(stack[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_roundtrip_simple() {
        let mut builder = DagBuilder::new();
        // A(B, C)
        let b = builder.insert_op(DagOp::Identifier("B".to_string()));
        let c = builder.insert_op(DagOp::Identifier("C".to_string()));
        let a = builder.insert_op(DagOp::CallFn {
            func_name: "A".to_string(),
            args: vec![b, c],
        });

        let seq = dag_to_sequence(&builder, a);
        assert_eq!(seq, vec!["CALL|A|2", "ARG|", "B", "ARG|", "C",]);

        let (mut new_builder, new_root) = sequence_to_dag(&seq).unwrap();
        assert_eq!(new_builder.nodes.len(), 3); // A, B, C
        assert_eq!(new_builder.nodes[new_root].op, builder.nodes[a].op);
    }

    #[test]
    fn test_sequence_roundtrip_hash_consing() {
        let mut builder = DagBuilder::new();
        // A(B, B) -> Duplicate should be merged during reconstruction
        let b = builder.insert_op(DagOp::Identifier("B".to_string()));
        let a = builder.insert_op(DagOp::CallFn {
            func_name: "A".to_string(),
            args: vec![b, b],
        });

        let seq = dag_to_sequence(&builder, a);
        assert_eq!(seq, vec!["CALL|A|2", "ARG|", "B", "ARG|", "B",]);

        let (new_builder, new_root) = sequence_to_dag(&seq).unwrap();
        // The newly constructed DAG should have exactly 2 nodes: B and A, because hash_consing merges B.
        assert_eq!(new_builder.nodes.len(), 2);

        let root_node = &new_builder.nodes[new_root];
        if let DagOp::CallFn { args, .. } = &root_node.op {
            assert_eq!(
                args[0], args[1],
                "Hash consing failed to merge duplicate B arguments"
            );
        } else {
            panic!("Root not CallFn");
        }
    }

    #[test]
    fn test_action_sequence_to_dag_logic() {
        let mut builder = DagBuilder::new();
        let seq = vec![
            "shift".to_string(),
            "shift".to_string(),
            "divide".to_string(),
        ];

        let param_names = vec!["b".to_string(), "a".to_string()]; // reversed
        let available_funcs = vec![(
            "divide".to_string(),
            vec![TypeDecl::Float, TypeDecl::Float],
            TypeDecl::Float,
        )];

        let root =
            action_sequence_to_dag(&seq, param_names, &available_funcs, &mut builder).unwrap();

        assert_eq!(builder.nodes.len(), 3); // a, b, divide
        let root_node = &builder.nodes[root];
        if let DagOp::CallFn { func_name, args } = &root_node.op {
            assert_eq!(func_name, "divide");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Root not CallFn");
        }
    }
}
