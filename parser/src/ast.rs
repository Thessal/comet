use std::fmt;

use crate::{behavior::BehaviorDecl, expr::Literal};
use stdlib::OperatorSpec;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub root: usize,
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
    Behavior(BehaviorDecl),
}

impl Network {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), root: 0 }
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

    pub fn extract_subtree(&self, node_id: usize) -> Network {
        let mut new_network = Network::new();

        fn copy_node(old_net: &Network, new_net: &mut Network, current_id: usize) -> usize {
            let node = &old_net.nodes[current_id];
            let node_type = node.node_type.clone();
            let children = node.children.clone();

            let new_id = new_net.add_node(node_type);
            for child_id in children {
                let new_child_id = copy_node(old_net, new_net, child_id);
                new_net.add_child(new_id, new_child_id);
            }
            new_id
        }

        let new_root = copy_node(self, &mut new_network, node_id);
        new_network.root = new_root;
        new_network
    }

    pub fn format_node(&self, node_id: usize) -> String {
        // Used to hash for caching
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
            NodeType::Behavior(behavior) => {
                let params: Vec<String> =
                    node.children.iter().map(|&c| self.format_node(c)).collect();
                format!(
                    "{}({})",
                    behavior.name.as_ref().unwrap_or(&"_".into()),
                    params.join(", ")
                )
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

    pub fn get_behavior(&self) -> (usize, &BehaviorDecl) {
        let behavior_indices = self.get_behavior_indices();
        assert!(
            behavior_indices.len() == 1,
            "Exactly one behavior node in AST is supported, currenlty."
        );
        let behavior_idx = behavior_indices[0];
        match &self.nodes[behavior_idx].node_type {
            NodeType::Behavior(b) => (behavior_idx, &b),
            _ => panic!(),
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            return write!(f, "Empty Network");
        }
        write!(f, "{}", self.format_node(self.root))
    }
}

#[cfg(test)]
mod tests {
    use stdlib::types::Signal;

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

        let behavior = BehaviorDecl::new(
            "Mix",
            vec![Signal::DataFrame(None), Signal::DataFrame(None)],
            Signal::DataFrame(None),
        );
        let mixed = network.add_node(NodeType::Behavior(behavior));
        network.add_child(mixed, op1);
        network.add_child(mixed, op2);

        let lit3 = network.add_node(NodeType::Literal(Literal::Float(2.0)));
        let root = network.add_node(NodeType::Operator(OperatorSpec::from("divide")));
        network.root = root;
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
