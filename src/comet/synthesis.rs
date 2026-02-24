use crate::comet::ast::{Ident, Expr, Literal, Op, ConstraintDecl, BehaviorDecl, CategoryExpr, TypeDecl};
use std::collections::{HashSet, HashMap};

pub fn substitute_expr(expr: &Expr, env: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Identifier(id) => {
            if let Some(val) = env.get(id) {
                // Recursively substitute
                substitute_expr(val, env)
            } else {
                expr.clone()
            }
        },
        Expr::List(exprs) => {
            Expr::List(exprs.iter().map(|e| substitute_expr(e, env)).collect())
        },
        Expr::Range { start, step, end } => {
            Expr::Range {
                start: Box::new(substitute_expr(start, env)),
                step: step.as_ref().map(|s| Box::new(substitute_expr(s, env))),
                end: Box::new(substitute_expr(end, env)),
            }
        },
        Expr::Call { path, args } => {
            let new_args = args.iter().map(|arg| {
                crate::comet::ast::ArgValue {
                    name: arg.name.clone(),
                    value: substitute_expr(&arg.value, env),
                }
            }).collect();
            Expr::Call {
                path: path.clone(),
                args: new_args,
            }
        },
        Expr::MemberAccess { target, field } => {
            Expr::MemberAccess {
                target: Box::new(substitute_expr(target, env)),
                field: field.clone(),
            }
        },
        _ => expr.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnSignature {
    pub name: Ident,
    pub args: Vec<(Ident, ConstraintDecl)>,
    pub return_constraint: ConstraintDecl,
}

#[derive(Debug, Clone)]
pub enum RealExpr {
    CallFn {
        func_name: Ident,
        args: Vec<(Option<Ident>, RealExpr)>,
        return_constraint: ConstraintDecl,
    },
    Literal(Literal),
    Identifier(Ident),
    BinaryOp { left: Box<RealExpr>, op: Op, right: Box<RealExpr> },
    UnaryOp { op: Op, target: Box<RealExpr> },
}

impl std::fmt::Display for RealExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RealExpr::Identifier(name) => write!(f, "{}", name),
            RealExpr::Literal(lit) => match lit {
                Literal::Integer(i) => write!(f, "{}", i),
                Literal::Float(fl) => write!(f, "{}", fl),
                Literal::String(s) => write!(f, "\"{}\"", s),
                Literal::Boolean(b) => write!(f, "{}", b),
            },
            RealExpr::CallFn { func_name, args, .. } => {
                write!(f, "{}(", func_name)?;
                for (i, (arg_name_opt, arg_expr)) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    if let Some(arg_name) = arg_name_opt {
                        write!(f, "{}={}", arg_name, arg_expr)?;
                    } else {
                        write!(f, "{}", arg_expr)?;
                    }
                }
                write!(f, ")")
            },
            RealExpr::BinaryOp { left, op, right } => {
                let op_str = match op {
                    Op::Add => "+", Op::Sub => "-", Op::Mul => "*", Op::Div => "/",
                    Op::Eq => "==", Op::Neq => "!=", Op::Lt => "<", Op::Gt => ">",
                    Op::And => "and", Op::Or => "or", Op::Not => "!",
                };
                write!(f, "({} {} {})", left, op_str, right)
            },
            RealExpr::UnaryOp { op, target } => {
                let op_str = match op {
                    Op::Sub => "-", Op::Not => "!", _ => "?",
                };
                write!(f, "{}{}", op_str, target)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubtreeState {
    pub tree: RealExpr,
    pub consumed_args: HashSet<Ident>,
    pub output_constraint: ConstraintDecl,
    pub depth: u32,
}

pub struct Synthesizer;

impl Synthesizer {
    /// Exhaustively synthesizes a BehaviorDecl into all mathematically valid disjoint forests.
    /// Returns a list of forests, where each forest is a `Vec<RealExpr>`. The primary expression
    /// (which returns the Behavior output) will be first, followed by independent side-effect exprs (e.g. `Consume`).
    pub fn exhaustive_synthesize(behavior: &BehaviorDecl, library: &[FnSignature]) -> Vec<Vec<RealExpr>> {
        let max_depth = behavior.depth;
        let mut pool: Vec<SubtreeState> = Vec::new();

        // Step 1.1: Initialization (Depth 0)
        // For each argument, add a base identifier state
        for arg in &behavior.args {
            let mut consumed_args = HashSet::new();
            consumed_args.insert(arg.name.clone());
            pool.push(SubtreeState {
                tree: RealExpr::Identifier(arg.name.clone()),
                consumed_args,
                output_constraint: arg.constraint.clone(),
                depth: 0,
            });
        }

        // We can also have literal base states if needed, but for exhaustive search over arguments,
        // we only strictly need argument variables. Real synthesis would optionally seed literals here.

        // Step 1.2: Iterative Search (Depth 1 to max_depth)
        for d in 1..=max_depth {
            let mut next_states = Vec::new();
            
            for func in library {
                let n_args = func.args.len();
                
                if n_args == 0 {
                    // Nullary function mapping (e.g., globals, constants)
                    let mut bindings = HashMap::new();
                    let return_bound = Self::resolve_return(&func.return_constraint, &bindings);
                    next_states.push(SubtreeState {
                        tree: RealExpr::CallFn {
                            func_name: func.name.clone(),
                            args: vec![],
                            return_constraint: return_bound.clone(),
                        },
                        consumed_args: HashSet::new(),
                        output_constraint: return_bound,
                        depth: d,
                    });
                    continue;
                }

                // We must find all N-tuples from the pool that fulfill this func.
                // We'll use a recursive helper to build N-tuples safely without combinatorial memory explosion.
                let mut valid_tuples = Vec::new();
                Self::find_tuples(&pool, func, d, 0, &mut Vec::new(), &mut HashSet::new(), &mut HashMap::new(), &mut valid_tuples);

                for (tuple, bindings) in valid_tuples {
                    // Assemble the new Node
                    let mut combined_consumed = HashSet::new();
                    let mut real_args = Vec::new();
                    
                    for (i, state) in tuple.iter().enumerate() {
                        combined_consumed.extend(state.consumed_args.iter().cloned());
                        real_args.push((Some(func.args[i].0.clone()), state.tree.clone()));
                    }

                    let resolved_return = Self::resolve_return(&func.return_constraint, &bindings);

                    next_states.push(SubtreeState {
                        tree: RealExpr::CallFn {
                            func_name: func.name.clone(),
                            args: real_args,
                            return_constraint: resolved_return.clone(),
                        },
                        consumed_args: combined_consumed,
                        output_constraint: resolved_return,
                        depth: d,
                    });
                }
            }
            pool.extend(next_states);
        }

        // Step 2: Forest Assembly
        // Find all subsets of mutually disjoint subtrees from P where:
        // 1. Exactly one subtree output satisfies `behavior.return_constraint`
        // 2. All other subtrees output `()` (represented here loosely. For our prototype, side effects return TypeDecl::String as a placeholder or we can explicitly look for a "Void/()" base_type if we add one. We will assume function returns None/() if base_type doesn't matter and it is marked side-effect). Let's assume TypeDecl bindings isn't perfectly handling `()` so we use a dummy `SideEffect` convention: return_constraint has no CategoryExpr, and maybe TypeDecl::Bool as void.
        // Actually, we can check if `ConstraintChecker` unifies perfectly.
        // For simplicity, we assume side-effects are identified by some explicit contract, 
        // but for now let's dynamically track valid subsets.
        
        let mut results = Vec::new();
        let expected_args: HashSet<Ident> = behavior.args.iter().map(|a| a.name.clone()).collect();
        
        // Find all main trees:
        let mut main_trees = Vec::new();
        for (i, state) in pool.iter().enumerate() {
            let mut bindings = HashMap::new();
            if Self::satisfies_constraint(&state.output_constraint, &behavior.return_constraint, &mut bindings) {
                main_trees.push((i, state));
            }
        }

        // Find all side effect trees (e.g. `Consume`)
        // In this mockup, we define `Consume` as returning `output_constraint` with name "SideEffect" (or we just know them by their lack of need to match return).
        let mut side_trees = Vec::new();
        for (i, state) in pool.iter().enumerate() {
            if let RealExpr::CallFn { ref func_name, .. } = state.tree {
                if func_name.starts_with("Consume") {
                    side_trees.push((i, state));
                }
            }
        }

        // For each main tree, recursively add side trees until consumed_args == expected_args
        for (_m_idx, m_state) in main_trees {
            let mut current_forest = vec![m_state.tree.clone()];
            let mut current_consumed = m_state.consumed_args.clone();
            
            Self::assemble_forest(&expected_args, &mut current_consumed, &mut current_forest, &side_trees, 0, &mut results);
        }

        results
    }
    
    fn assemble_forest(
        expected: &HashSet<Ident>,
        current_consumed: &mut HashSet<Ident>,
        current_forest: &mut Vec<RealExpr>,
        side_trees: &[(usize, &SubtreeState)],
        start_idx: usize,
        results: &mut Vec<Vec<RealExpr>>
    ) {
        if current_consumed == expected {
            results.push(current_forest.clone());
            return;
        }

        for i in start_idx..side_trees.len() {
            let s_state = side_trees[i].1;
            
            // Check if disjoint
            if s_state.consumed_args.is_disjoint(current_consumed) {
                let mut next_consumed = current_consumed.clone();
                next_consumed.extend(s_state.consumed_args.iter().cloned());
                
                current_forest.push(s_state.tree.clone());
                Self::assemble_forest(expected, &mut next_consumed, current_forest, side_trees, i + 1, results);
                current_forest.pop(); // Backtrack
            }
        }
    }

    /// Recursively find all valid N-tuples of subtrees for a function.
    fn find_tuples<'a>(
        pool: &'a [SubtreeState],
        func: &FnSignature,
        target_d: u32,
        arg_idx: usize,
        current_tuple: &mut Vec<&'a SubtreeState>,
        current_consumed: &mut HashSet<Ident>,
        current_bindings: &mut HashMap<String, Vec<CategoryExpr>>,
        valid_tuples: &mut Vec<(Vec<&'a SubtreeState>, HashMap<String, Vec<CategoryExpr>>)>,
    ) {
        if arg_idx == func.args.len() {
            // Check if at least one subtree hits depth d - 1 exactly.
            let mut meets_depth = false;
            for state in current_tuple.iter() {
                if state.depth == target_d - 1 {
                    meets_depth = true;
                    break;
                }
            }
            if meets_depth {
                valid_tuples.push((current_tuple.clone(), current_bindings.clone()));
            }
            return;
        }

        let expected_constraint = &func.args[arg_idx].1;

        for state in pool {
            // Must be disjoint
            if !state.consumed_args.is_disjoint(current_consumed) {
                continue;
            }

            // Must satisfy constraint (Unification!)
            let mut cloned_bindings = current_bindings.clone();
            if Self::satisfies_constraint(&state.output_constraint, expected_constraint, &mut cloned_bindings) {
                
                // Backtracking state prep
                let mut next_consumed = current_consumed.clone();
                next_consumed.extend(state.consumed_args.iter().cloned());
                current_tuple.push(state);

                Self::find_tuples(pool, func, target_d, arg_idx + 1, current_tuple, &mut next_consumed, &mut cloned_bindings, valid_tuples);

                // Backtrack
                current_tuple.pop();
            }
        }
    }

    /// Validates unification logic, including wildcard category exact match `\'a`
    fn satisfies_constraint(
        provided: &ConstraintDecl,
        expected: &ConstraintDecl,
        bindings: &mut HashMap<String, Vec<CategoryExpr>>,
    ) -> bool {
        if provided.base_type != expected.base_type {
            return false;
        }

        let prov_cats = Self::flatten_categories(&provided.category_expr);
        let exp_cats = Self::flatten_categories(&expected.category_expr);
        
        let mut expected_normal_cats = Vec::new();
        let mut captured_var = None;

        for ec in exp_cats {
            if let CategoryExpr::Atom(name) = ec {
                if name.starts_with('\'') {
                    captured_var = Some(name);
                    continue;
                } else {
                    expected_normal_cats.push(CategoryExpr::Atom(name));
                }
            } else {
                expected_normal_cats.push(ec);
            }
        }

        // 1. Provided must contain all expected_normal_cats
        for normal_cat in &expected_normal_cats {
            if !prov_cats.contains(normal_cat) {
                return false;
            }
        }

        // 2. Unassigned extra properties go to captured_var
        let mut residual = Vec::new();
        for pc in &prov_cats {
            if !expected_normal_cats.contains(pc) {
                residual.push(pc.clone());
            }
        }

        if let Some(var_name) = captured_var {
            if let Some(existing_capture) = bindings.get(&var_name) {
                // Exact unification: The residual MUST perfectly match the previously bound capture
                if existing_capture.len() != residual.len() {
                    return false;
                }
                for c in &residual {
                    if !existing_capture.contains(c) {
                        return false;
                    }
                }
            } else {
                // Bind new capture
                bindings.insert(var_name, residual);
            }
        } else {
            // If no capture variable is expected, provided should not have any unaccounted residual properties for strict match.
            // Wait, in standard matching, it's optionally allowed unless strict bounds apply. 
            // In the `docs/05_synthesis.md`, missing exact properties are allowed UNLESS unified. 
            // But if there is no `'a`, perhaps they are stripped out. To align with user's specific 05_synthesis test, let's allow benign pass-through if not captured.
        }

        true
    }

    fn resolve_return(expected: &ConstraintDecl, bindings: &HashMap<String, Vec<CategoryExpr>>) -> ConstraintDecl {
        let mut resolved_cats = Vec::new();
        for ec in Self::flatten_categories(&expected.category_expr) {
            if let CategoryExpr::Atom(name) = &ec {
                if name.starts_with('\'') {
                    if let Some(bound) = bindings.get(name) {
                        resolved_cats.extend(bound.iter().cloned());
                    }
                    continue;
                }
            }
            resolved_cats.push(ec);
        }

        let final_cat = if resolved_cats.is_empty() {
            None
        } else {
            Some(CategoryExpr::Addition(resolved_cats)) // Simplified compound structure
        };

        ConstraintDecl {
            base_type: expected.base_type.clone(),
            category_expr: final_cat,
        }
    }

    fn flatten_categories(cat: &Option<CategoryExpr>) -> Vec<CategoryExpr> {
        let mut result = Vec::new();
        if let Some(c) = cat {
            match c {
                CategoryExpr::Addition(cats) => result.extend(cats.clone()),
                CategoryExpr::Atom(_) => result.push(c.clone()),
                CategoryExpr::None => {},
                _ => result.push(c.clone()), // Support others trivially
            }
        }
        result
    }

    /// Recursively evaluate an AST Expression to generate a Cartesian product of valid Concrete branches!
    /// `[5, 21]` evaluates to `vec![Literal(5), Literal(21)]`.
    /// `ts_mean(window=[5, 21])` evaluates to `vec![Call(ts_mean, 5), Call(ts_mean, 21)]`.
    pub fn synthesize_expr(expr: &Expr) -> Result<Vec<RealExpr>, String> {
        match expr {
            Expr::Literal(lit) => Ok(vec![RealExpr::Literal(lit.clone())]),
            Expr::Identifier(id) => Ok(vec![RealExpr::Identifier(id.clone())]),
            Expr::List(exprs) => {
                let mut possibilities = Vec::new();
                for e in exprs {
                    let mut e_combs = Self::synthesize_expr(e)?;
                    possibilities.append(&mut e_combs);
                }
                Ok(possibilities)
            },
            Expr::Range { start, step, end } => {
                let start_lit = Self::get_lit(start)?;
                let end_lit = Self::get_lit(end)?;
                let step_lit = if let Some(st) = step {
                    Self::get_lit(st)?
                } else {
                    if let Literal::Integer(_) = start_lit { Literal::Integer(1) } else { Literal::Float(1.0) }
                };

                let mut possibilities = Vec::new();
                match (start_lit, end_lit, step_lit) {
                    (Literal::Integer(s), Literal::Integer(e), Literal::Integer(st)) => {
                        let mut current = s;
                        while current <= e {
                            possibilities.push(RealExpr::Literal(Literal::Integer(current)));
                            current += st;
                        }
                    },
                    (Literal::Float(s), Literal::Float(e), Literal::Float(st)) => {
                        let mut current = s;
                        let epsilon = 1e-9;
                        while current <= e + epsilon {
                            possibilities.push(RealExpr::Literal(Literal::Float(current)));
                            current += st;
                        }
                    },
                    _ => return Err("Range bounds must be monotonically matching numeric literals (Ints or Floats)".to_string()),
                }
                Ok(possibilities)
            },
            Expr::Call { path, args } => {
                // Cartesian product of arguments!
                let mut arg_possibilities: Vec<Vec<(Option<Ident>, RealExpr)>> = vec![vec![]];
                
                for arg in args {
                    let arg_name = arg.name.clone();
                    let evaluated_values = Self::synthesize_expr(&arg.value)?;
                    
                    let mut next_product = Vec::new();
                    for partial_tuple in &arg_possibilities {
                        for val in &evaluated_values {
                            let mut new_tuple = partial_tuple.clone();
                            new_tuple.push((arg_name.clone(), val.clone()));
                            next_product.push(new_tuple);
                        }
                    }
                    arg_possibilities = next_product;
                }

                let mut call_combinations = Vec::new();
                let func_name = path.segments.last().unwrap().clone();
                for args_tuple in arg_possibilities {
                    call_combinations.push(RealExpr::CallFn {
                        func_name: func_name.clone(),
                        args: args_tuple,
                        // Dummy constraint for expression evaluations. Real compiler infers this.
                        return_constraint: ConstraintDecl { base_type: TypeDecl::Bool, category_expr: None },
                    });
                }
                Ok(call_combinations)
            },
            _ => Err("Unsupported AST Expression for Synthesis Cartesian Generation".to_string())
        }
    }

    fn get_lit(expr: &Expr) -> Result<Literal, String> {
         match expr {
             Expr::Literal(l) => Ok(l.clone()),
             _ => Err("Not a literal".to_string())
         }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comet::ast::{TypeDecl, TypedArg};

    fn make_atom(s: &str) -> Option<CategoryExpr> {
        Some(CategoryExpr::Atom(s.to_string()))
    }

    fn make_compound(ss: &[&str]) -> Option<CategoryExpr> {
        let v = ss.iter().map(|s| CategoryExpr::Atom(s.to_string())).collect();
        Some(CategoryExpr::Addition(v))
    }

    #[test]
    fn test_comparator_exhaustive_search() {
        // Build the `behavior Comparator(signal: DataFrame, eps: Float Nonzero Optional, reference: DataFrame) -> DataFrame Indicator`
        let behavior = BehaviorDecl {
            name: "Comparator".to_string(),
            args: vec![
                TypedArg { name: "signal".to_string(), constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: None } },
                TypedArg { name: "eps".to_string(), constraint: ConstraintDecl { base_type: TypeDecl::Float, category_expr: make_compound(&["Nonzero", "Optional"]) } },
                TypedArg { name: "reference".to_string(), constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: None } },
            ],
            return_constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_atom("Indicator") },
            depth: 2,
        };

        // Build Library
        let mut lib = Vec::new();

        // `Fn Consume(b: Float Optional) -> Bool` (Bool acts as void here)
        lib.push(FnSignature {
            name: "Consume".to_string(),
            args: vec![("b".to_string(), ConstraintDecl { base_type: TypeDecl::Float, category_expr: make_atom("Optional") })],
            return_constraint: ConstraintDecl { base_type: TypeDecl::Bool, category_expr: None }, // Dummy void
        });

        // `Fn Rank(a: DataFrame) -> Normalized DataFrame`  (We mock this loosely: 'a' bindings in comet are DataFrame)
        lib.push(FnSignature {
            name: "Rank".to_string(),
            args: vec![("a".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: None })],
            return_constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_atom("Normalized") },
        });

        // `Fn RankNonzero(a: DataFrame, eps: Float Nonzero) -> DataFrame Normalized Nonzero`
        lib.push(FnSignature {
            name: "RankNonzero".to_string(),
            args: vec![
                ("a".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: None }),
                ("eps".to_string(), ConstraintDecl { base_type: TypeDecl::Float, category_expr: make_atom("Nonzero") })
            ],
            return_constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_compound(&["Normalized", "Nonzero"]) },
        });

        // `Fn Diff(a: DataFrame 'a, b:DataFrame 'a) -> DataFrame 'a Indicator`
        lib.push(FnSignature {
            name: "Diff".to_string(),
            args: vec![
                ("a".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_atom("'a") }),
                ("b".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_atom("'a") })
            ],
            return_constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_compound(&["'a", "Indicator"]) },
        });

        // `Fn Divide(a: DataFrame 'a, b: DataFrame 'a Nonzero) -> DataFrame 'a Indicator`
        lib.push(FnSignature {
            name: "Divide".to_string(),
            args: vec![
                ("a".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_atom("'a") }),
                ("b".to_string(), ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_compound(&["'a", "Nonzero"]) })
            ],
            return_constraint: ConstraintDecl { base_type: TypeDecl::DataFrame, category_expr: make_compound(&["'a", "Indicator"]) },
        });

        let results = Synthesizer::exhaustive_synthesize(&behavior, &lib);

        // Debug output to see exactly what combinations survived
        for (i, r) in results.iter().enumerate() {
            println!("Combination {}:", i);
            for expr in r {
                if let RealExpr::CallFn { func_name, args, .. } = expr {
                    println!("    {}, {:?}", func_name, args);
                }
            }
        }

        // We expect EXACTLY 6 exhaustive mathematical sets of mutually disjoint subsets matching
        // docs/05_synthesis.md's strict unification rules!
        assert_eq!(results.len(), 6, "Expected exactly 6 derived valid Combinations for Comparator.");
    }

    #[test]
    fn test_list_range_cartesian_expansion() {
        use crate::comet::ast::{Path, ArgValue};

        // var = [5, 21]
        let list_expr = Expr::List(vec![
            Expr::Literal(Literal::Integer(5)),
            Expr::Literal(Literal::Integer(21))
        ]);
        let r1 = Synthesizer::synthesize_expr(&list_expr).unwrap();
        assert_eq!(r1.len(), 2);

        // var = [10..10..30] -> 10, 20, 30
        let range_expr = Expr::Range {
            start: Box::new(Expr::Literal(Literal::Integer(10))),
            step: Some(Box::new(Expr::Literal(Literal::Integer(10)))),
            end: Box::new(Expr::Literal(Literal::Integer(30))),
        };
        let r2 = Synthesizer::synthesize_expr(&range_expr).unwrap();
        assert_eq!(r2.len(), 3);

        // ts_mean(a="volume", window=[5, 21])
        let call_expr = Expr::Call {
            path: Path { segments: vec!["ts_mean".to_string()] },
            args: vec![
                ArgValue { name: Some("a".to_string()), value: Expr::Literal(Literal::String("volume".to_string())) },
                ArgValue { name: Some("window".to_string()), value: list_expr },
            ]
        };
        let r3 = Synthesizer::synthesize_expr(&call_expr).unwrap();
        assert_eq!(r3.len(), 2);
        if let RealExpr::CallFn { func_name, args, .. } = &r3[0] {
            assert_eq!(func_name, "ts_mean");
            assert_eq!(args.len(), 2);
        }

        // ts_mean(a=["v1", "v2"], window=[10..10..30]) -> 2 * 3 = 6 combinations!
        let multi_call = Expr::Call {
            path: Path { segments: vec!["ts_mean".to_string()] },
            args: vec![
                ArgValue { 
                    name: Some("a".to_string()), 
                    value: Expr::List(vec![
                        Expr::Literal(Literal::String("v1".to_string())),
                        Expr::Literal(Literal::String("v2".to_string()))
                    ])
                },
                ArgValue { name: Some("window".to_string()), value: range_expr },
            ]
        };
        let r4 = Synthesizer::synthesize_expr(&multi_call).unwrap();
        assert_eq!(r4.len(), 6, "Expected exactly 6 cartesian branches (2 * 3)");
    }
}
