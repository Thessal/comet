use crate::comet::ast::Constraint;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Atom {
    Type(String),
    Variable(String), // 'a
}

// Expanded representation of a type
// A type is a Set of "Constraint Chains"
// Series NonZero -> { {Series, NonZero} }
// Series | DataFrame -> { {Series}, {DataFrame} }
pub type ConstraintSet = HashSet<Vec<Atom>>;

pub fn expand(constraint: &Constraint) -> ConstraintSet {
    match constraint {
        Constraint::Atom(name) => {
            let mut set = HashSet::new();
            if name.starts_with("'") {
                 set.insert(vec![Atom::Variable(name.clone())]);
            } else {
                 set.insert(vec![Atom::Type(name.clone())]);
            }
            set
        },
        Constraint::Addition(constraints) => {
            // Intersection / Combination
            // (A | B) C -> A C | B C
            // Start with { [] } (identity)
            let mut result: ConstraintSet = HashSet::new();
            result.insert(vec![]);
            
            for c in constraints {
                let expanded_c = expand(c);
                let mut next_result = HashSet::new();
                for existing in &result {
                    for incoming in &expanded_c {
                         let mut combined = existing.clone();
                         // Add incoming atoms, avoiding duplicates (set-like behavior for list)
                         for atom in incoming {
                             if !combined.contains(atom) {
                                 combined.push(atom.clone());
                             }
                         }
                         // Sort for canonical representation
                         combined.sort_by(|a, b| match (a, b) {
                             (Atom::Type(s1), Atom::Type(s2)) => s1.cmp(s2),
                             (Atom::Variable(s1), Atom::Variable(s2)) => s1.cmp(s2),
                             (Atom::Type(_), Atom::Variable(_)) => std::cmp::Ordering::Less,
                             (Atom::Variable(_), Atom::Type(_)) => std::cmp::Ordering::Greater,
                         });
                         next_result.insert(combined);
                    }
                }
                result = next_result;
            }
            result
        },
        Constraint::Union(constraints) => {
            let mut result = HashSet::new();
            for c in constraints {
                let expanded = expand(c);
                for chain in expanded {
                    result.insert(chain);
                }
            }
            result
        },
        Constraint::Subtraction(lhs, rhs) => {
             let left_set = expand(lhs);
             // Logic: Remove ANY chain from left that "matches" rhs logic? 
             // Spec: (Series | DataFrame) - DataFrame -> Series
             // Matches(chain, constraint) -> bool
             // But we need to be careful. Subtraction in spec seems to operate on the expansion.
             // If a chain in Left MATCHES Right, remove it.
             
             let mut result = HashSet::new();
             for chain in left_set {
                 if !matches_chain(&chain, rhs) {
                     result.insert(chain);
                 }
             }
             result
        },
        Constraint::None => HashSet::new(),
    }
}

// Check if a specific chain (Type instance) matches a constraint
pub fn matches_chain(chain: &Vec<Atom>, constraint: &Constraint) -> bool {
    let expanded_constraint = expand(constraint);
    // Matches if the chain is a SUPERSET of any chain in expanded_constraint?
    // Ex: Chain {Series, NonZero} matches Constraint {Series} ? Yes.
    // Ex: Chain {Series} matches Constraint {Series, NonZero} ? No.
    // Ex: Chain {Series, NonZero} matches Constraint {Series | DataFrame} ? Yes (matches Series).
    
    for req_chain in expanded_constraint {
        // req_chain must be subset of chain
        let mut all_found = true;
        for req_atom in req_chain {
            if !chain.contains(&req_atom) {
                all_found = false;
                break;
            }
        }
        if all_found {
            return true;
        }
    }
    false
}
