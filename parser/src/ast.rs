use std::fmt;

use crate::expr::Literal;
use stdlib::OperatorSpec;
use stdlib::types::Signal;

//////////////
/* AST Node */
//////////////

#[derive(Debug, Clone, PartialEq)]
pub enum Tree {
    Operator(OperatorNode),
    Literal(Literal),
    Behavior(BehaviorNode),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorNode {
    pub name: String,
    pub parameters: Option<Vec<Tree>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OperatorNode {
    pub spec: OperatorSpec,
    pub parameters: Option<Vec<Tree>>,
}

impl OperatorNode {
    pub fn new(operator_name: &str, args: Vec<Tree>) -> Self {
        let spec = OperatorSpec::from(operator_name);
        OperatorNode {
            spec,
            parameters: Some(args),
        }
    }
}

#[cfg(test)]
mod tests {
    // #[test]
    // fn test() {
    // }
}
