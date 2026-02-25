

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
    pub ast_string: String,
    pub primary_output_node: usize,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        ExecutionGraph { nodes: Vec::new(), ast_string: String::new(), primary_output_node: 0 }
    }

    pub fn add_node(&mut self, node: ExecutionNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    pub fn from_real_expr(expr: &crate::comet::synthesis::RealExpr) -> Self {
        let mut graph = ExecutionGraph::new();
        graph.ast_string = expr.to_string();
        graph.primary_output_node = Self::build_graph(&mut graph, expr);
        graph
    }

    pub fn from_forest(forest: &[crate::comet::synthesis::RealExpr]) -> Self {
        let mut graph = ExecutionGraph::new();
        let s: Vec<String> = forest.iter().map(|t| t.to_string()).collect();
        graph.ast_string = s.join("; ");
        for (i, tree) in forest.iter().enumerate() {
            let root_id = Self::build_graph(&mut graph, tree);
            if i == 0 {
                graph.primary_output_node = root_id;
            }
        }
        graph
    }

    fn build_graph(graph: &mut ExecutionGraph, expr: &crate::comet::synthesis::RealExpr) -> usize {
        match expr {
            crate::comet::synthesis::RealExpr::Identifier(name) => {
                graph.add_node(ExecutionNode::Source {
                    name: name.clone(),
                    type_name: "TimeSeries".to_string(), // Simplified default
                })
            },
            crate::comet::synthesis::RealExpr::Literal(lit) => {
                let cv = match lit {
                    crate::comet::ast::Literal::Integer(i) => ConstantValue::Integer(*i),
                    crate::comet::ast::Literal::Float(f) => ConstantValue::Float(*f),
                    crate::comet::ast::Literal::String(_) => ConstantValue::String,
                    crate::comet::ast::Literal::Boolean(_) => ConstantValue::Boolean,
                };
                graph.add_node(ExecutionNode::Constant { value: cv })
            },
            crate::comet::synthesis::RealExpr::CallFn { func_name, args, .. } => {
                if func_name == "data" {
                    if let Some((_, arg_expr)) = args.first() {
                        if let crate::comet::synthesis::RealExpr::Literal(crate::comet::ast::Literal::String(s)) = arg_expr {
                            return graph.add_node(ExecutionNode::Source {
                                name: s.clone(),
                                type_name: "DataFrame".to_string(), // Keep consistent
                            });
                        }
                    }
                }
                let mut arg_indices = Vec::new();
                for (_, arg_expr) in args {
                    arg_indices.push(Self::build_graph(graph, arg_expr));
                }
                graph.add_node(ExecutionNode::Operation {
                    op: func_name.clone(),
                    args: arg_indices,
                })
            },

        }
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
