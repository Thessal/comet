use std::fmt;

use crate::expr::Literal;
use stdlib::OperatorSpec;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Network {
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub node_type: NodeType,
    pub children: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Operator(OperatorSpec),
    Literal(Literal),
    Behavior(String),
}

impl Network {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node_type: NodeType) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            node_type,
            children: Vec::new(),
        });
        index
    }

    pub fn add_child(&mut self, parent: usize, child: usize) {
        self.nodes[parent].children.push(child);
    }

    pub fn format_node(&self, node_id: usize) -> String {
        if node_id >= self.nodes.len() {
            return String::new();
        }
        let node = &self.nodes[node_id];
        match &node.node_type {
            NodeType::Operator(op) => {
                let params: Vec<String> =
                    node.children.iter().map(|&c| self.format_node(c)).collect();
                format!("{}({})", op.name, params.join(", "))
            }
            NodeType::Behavior(name) => {
                let params: Vec<String> =
                    node.children.iter().map(|&c| self.format_node(c)).collect();
                format!("{}({})", name, params.join(", "))
            }
            NodeType::Literal(lit) => {
                format!("{}", lit)
            }
        }
    }

    pub fn get_behavior_indices(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                if let NodeType::Behavior(_) = n.node_type {
                    true
                } else {
                    false
                }
            })
            .map(|(i, _)| i)
            .collect()
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            return write!(f, "Empty Network");
        }
        write!(f, "{}", self.format_node(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network() {
        let mut network = Network::new();

        // 1. build a ast that contains a behavior node
        // divide(Mix(data("volume"), data("adv20")), 2) where Mix is a behavior.

        let lit1 = network.add_node(NodeType::Literal(Literal::String("volume".to_string())));
        let op1 = network.add_node(NodeType::Operator(OperatorSpec::from("data")));
        network.add_child(op1, lit1);

        let lit2 = network.add_node(NodeType::Literal(Literal::String("adv20".to_string())));
        let op2 = network.add_node(NodeType::Operator(OperatorSpec::from("data")));
        network.add_child(op2, lit2);

        let mixed = network.add_node(NodeType::Behavior("Mix".to_string()));
        network.add_child(mixed, op1);
        network.add_child(mixed, op2);

        let lit3 = network.add_node(NodeType::Literal(Literal::Float(2.0)));
        let root = network.add_node(NodeType::Operator(OperatorSpec::from("divide")));
        network.add_child(root, mixed);
        network.add_child(root, lit3);

        let display_str = network.format_node(root);
        assert_eq!(
            display_str,
            format!(
                "divide(Mix(data(\"volume\"), data(\"adv20\")), {})",
                Literal::Float(2.0)
            )
        );

        // 2. Replace Mix with add operator.
        let behavior_idx = mixed;
        let behavior_node = &mut network.nodes[behavior_idx];
        behavior_node.node_type = NodeType::Operator(OperatorSpec::from("add"));

        // 3. Display divide(divide(data("volume"), data("adv20")), 2)
        let display_str = network.format_node(root);
        assert_eq!(
            display_str,
            format!(
                "divide(add(data(\"volume\"), data(\"adv20\")), {})",
                Literal::Float(2.0)
            )
        );
        println!("{}", display_str);
    }
}
