

#[derive(Debug, Clone)]
pub enum ConstantValue {
    Integer(i64),
    Float(f64),
    String,
    Boolean,
}

#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    pub nodes: Vec<ExecutionNode>,
    pub ast_strings: Vec<String>,
    pub output_nodes: Vec<usize>,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        ExecutionGraph { nodes: Vec::new(), ast_strings: Vec::new(), output_nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: ExecutionNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn from_variants(variants: &[Vec<crate::comet::synthesis::RealExpr>]) -> Self {
        let mut graph = ExecutionGraph::new();
        let mut builder = crate::comet::dag::DagBuilder::new();
        
        let mut variant_roots = Vec::new();
        for variant in variants {
            let s: Vec<String> = variant.iter().map(|t| t.to_string()).collect();
            graph.ast_strings.push(s.join("; "));
            
            let roots = builder.build_from_forest(variant);
            if let Some(&root_id) = roots.first() {
                variant_roots.push(root_id);
            }
        }
        
        let mut dag_to_graph_id = std::collections::HashMap::new();
        
        for node in &builder.nodes {
            let eq_node = match &node.op {
                crate::comet::dag::DagOp::Literal(lit) => {
                    let cv = match lit {
                        crate::comet::ast::Literal::Integer(i) => ConstantValue::Integer(*i),
                        crate::comet::ast::Literal::Float(f) => ConstantValue::Float(*f),
                        crate::comet::ast::Literal::String(_) => ConstantValue::String,
                        crate::comet::ast::Literal::Boolean(_) => ConstantValue::Boolean,
                    };
                    ExecutionNode::Constant { value: cv }
                },
                crate::comet::dag::DagOp::Identifier(name) => {
                    ExecutionNode::Source {
                        name: name.clone(),
                        type_name: "TimeSeries".to_string(),
                    }
                },
                crate::comet::dag::DagOp::CallFn { func_name, args } => {
                    if func_name == "data" {
                        let mut src_name = "unknown".to_string();
                        if let Some((_, arg_id)) = args.first() {
                            if let crate::comet::dag::DagOp::Literal(crate::comet::ast::Literal::String(s)) = &builder.nodes[*arg_id].op {
                                src_name = s.clone();
                            }
                        }
                        ExecutionNode::Source {
                            name: src_name,
                            type_name: "DataFrame".to_string(),
                        }
                    } else {
                        let mut exec_args = Vec::new();
                        for (_, arg_id) in args {
                            let mapped_id = *dag_to_graph_id.get(arg_id).expect("DAG topological sort failed");
                            exec_args.push(mapped_id);
                        }
                        ExecutionNode::Operation {
                            op: func_name.clone(),
                            args: exec_args,
                        }
                    }
                }
            };
            
            let id = graph.add_node(eq_node);
            dag_to_graph_id.insert(node.id, id);
        }
        
        for root_id in variant_roots {
             graph.output_nodes.push(*dag_to_graph_id.get(&root_id).unwrap_or(&0));
        }
        
        graph
    }
}

#[derive(Debug, Clone)]
pub enum ExecutionNode {
    Source {
        name: String,
        type_name: String,
    },
    Constant {
        value: ConstantValue,
    },
    Operation {
        op: String,
        args: Vec<usize>, // Indices into nodes
    },
}
