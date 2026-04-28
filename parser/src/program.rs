// input code AST
use crate::expr::Ident;
use crate::expr::Stmt;
use stdlib::types::Signal;
pub type InputCode = Vec<InputDecl>;
pub type NamedSignal = (String, Signal);

#[derive(Debug, Clone, PartialEq)]
pub enum InputDecl {
    Import(String),
    Behavior(BehaviorDecl),
    Flow(FlowDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BehaviorDecl {
    pub inputs: Vec<NamedSignal>,
    pub output: NamedSignal,

    pub operators: Option<Vec<Ident>>,
    pub integers: Option<Vec<i64>>,
    pub floats: Option<Vec<f64>>,
    pub strings: Option<Vec<String>>,

    pub weights: Option<String>,
    pub train: Option<bool>,
    pub supervised_epochs: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlowDecl {
    pub name: Ident,
    pub body: Vec<Stmt>,
}

use std::fmt;

impl fmt::Display for InputDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputDecl::Import(i) => write!(f, "{}", i),
            InputDecl::Behavior(b) => write!(f, "{:?}", b),
            InputDecl::Flow(flow) => write!(f, "{:?}", flow),
        }
    }
}

impl fmt::Display for BehaviorDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let args: Vec<String> = self
            .inputs
            .iter()
            .map(|(name, dtype)| format!("{}: {:?}", name, dtype))
            .collect();
        let mut props = Vec::new();
        if let Some(w) = &self.weights {
            props.push(format!("weights = \"{}\"", w));
        }
        if let Some(t) = self.train {
            props.push(format!("train = {}", t));
        }
        if let Some(ss) = self.supervised_epochs {
            props.push(format!("supervised_epochs = {}", ss));
        }
        if let Some(ops) = &self.operators {
            props.push(format!("operators = [{}]", ops.join(", ")));
        }
        if let Some(ints) = &self.integers {
            let s: Vec<String> = ints.iter().map(|i| i.to_string()).collect();
            props.push(format!("integers = [{}]", s.join(", ")));
        }
        if let Some(flts) = &self.floats {
            let s: Vec<String> = flts.iter().map(|f| f.to_string()).collect();
            props.push(format!("floats = [{}]", s.join(", ")));
        }
        if let Some(strs) = &self.strings {
            let s: Vec<String> = strs.iter().map(|s| format!("\"{}\"", s)).collect();
            props.push(format!("strings = [{}]", s.join(", ")));
        }

        if props.is_empty() {
            writeln!(
                f,
                "Behavior {}({}) -> {:?}",
                self.output.0,
                args.join(", "),
                self.output.1
            )
        } else {
            writeln!(
                f,
                "Behavior {}({}) {{ {} }} ->  {:?}",
                self.output.0,
                args.join(", "),
                props.join(", "),
                self.output.1
            )
        }
    }
}

impl fmt::Display for FlowDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Flow {} {{", self.name)?;
        for stmt in &self.body {
            writeln!(f, "    {}", stmt)?;
        }
        writeln!(f, "}}")
    }
}
