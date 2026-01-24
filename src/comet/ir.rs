use crate::comet::ast::Ident;

#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    pub nodes: Vec<ExecutionNode>,
}

impl ExecutionGraph {
    pub fn new() -> Self {
        ExecutionGraph { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: ExecutionNode) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
}

#[derive(Debug, Clone)]
pub enum ExecutionNode {
    Source {
        name: String,
        type_name: String,
    },
    Constant {
        value: String,
        type_name: String,
    },
    Operation {
        op: OperatorOp,
        args: Vec<usize>, // Indices into nodes
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum OperatorOp {
    // Binary
    Divide,
    Multiply,
    Add, 
    Subtract,
    
    // Unary / Special
    Delay,
    Diff,
    
    // Rolling / Statistical
    RollingMean,
    RollingStd,
    ZScore,
    
    // Logic
    Filter,
    UpdateWhen,
    
    // Custom / Function Call used in synthesis
    FunctionCall(String), 
}
