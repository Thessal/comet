use crate::{
    behavior::*,
    expr::{Expr, FlowStmt, Stmt},
};
use pest::Parser;
use pest_derive::Parser;
use runtime::ast::Tree::{Literal, Program};
use stdlib::types::Signal;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct CometParser;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Pest error: {0}")]
    PestError(#[from] pest::error::Error<Rule>),
    #[error("Unexpected rule: {0:?}")]
    UnexpectedRule(Rule),
    #[error("Missing token")]
    MissingToken,
    #[error("Semantic error: {0}")]
    SemanticError(String),
}

pub fn parse(input: &str) -> Result<(FlowDecl, Vec<BehaviorDecl>, Vec<Program>), ParserError> {
    // Parses Flow and behavior.
    let mut pairs = CometParser::parse(Rule::program, input)?;
    let program_pair = pairs.next().ok_or(ParserError::MissingToken)?;
    let code: InputCode = parse_program(program_pair)?;
    let mut flow_opt = None;
    let behaviors: Vec<BehaviorDecl> = code
        .into_iter()
        .filter_map(|decl| match decl {
            InputDecl::Import(_) => None,
            InputDecl::Behavior(b) => Some(b),
            InputDecl::Flow(f) => {
                flow_opt = Some(f);
                None
            }
        })
        .collect();
    let flow = flow_opt.ok_or(ParserError::MissingToken)?;

    // Locates behavior call in the flow's body. (asserts that there's only one)
    // Locates assignments in the flow's body. Convert them into AST(Programs)
    // loop over flow.body until it meets FlowStmt::Expr. after it meet first Expr, stop loop and discard.
    // while looping, it should meet FlowStmt::Assignment
    // Store Assignments to assignments, and output to output.
    let mut assignments = Vec::new();
    let mut output = None;

    for stmt in flow.body.iter() {
        match stmt {
            FlowStmt::Assignment { target, expr } => {
                assignments.push((target.clone(), expr.clone()));
            }
            FlowStmt::Expr(expr) => {
                output = Some(expr.clone());
                break; // Stop loop after the first Expr is found
            }
        }
    }

    let target_call = extract_single_behavior_call(&assignments, &output, &behaviors)?;
    let ast_programs = build_ast_programs(target_call, &assignments)?;

    Ok((flow, behaviors, ast_programs))
}

fn extract_single_behavior_call<'a>(
    assignments: &'a [(crate::expr::Ident, Expr)],
    output: &'a Option<Expr>,
    behaviors: &[BehaviorDecl],
) -> Result<&'a Expr, ParserError> {
    let behavior_names: std::collections::HashSet<&str> = behaviors.iter().map(|b| b.output.0.as_str()).collect();
    let mut behavior_calls = Vec::new();

    fn extract_calls<'a>(expr: &'a Expr, names: &std::collections::HashSet<&str>, calls: &mut Vec<&'a Expr>) {
        match expr {
            Expr::Call { path, args } => {
                if let Some(name) = path.segments.first() {
                    if names.contains(name.as_str()) {
                        calls.push(expr);
                    }
                }
                for arg in args {
                    extract_calls(arg, names, calls);
                }
            }
            Expr::MemberAccess { target, .. } => extract_calls(target, names, calls),
            Expr::List(exprs) => {
                for e in exprs {
                    extract_calls(e, names, calls);
                }
            }
            Expr::Range { start, step, end } => {
                extract_calls(start, names, calls);
                if let Some(s) = step {
                    extract_calls(s, names, calls);
                }
                extract_calls(end, names, calls);
            }
            _ => {}
        }
    }

    for (_, expr) in assignments {
        extract_calls(expr, &behavior_names, &mut behavior_calls);
    }
    if let Some(out_expr) = output {
        extract_calls(out_expr, &behavior_names, &mut behavior_calls);
    }

    if behavior_calls.len() != 1 {
        return Err(ParserError::SemanticError(format!(
            "Expected exactly 1 behavior call, found {}",
            behavior_calls.len()
        )));
    }

    Ok(behavior_calls[0])
}

fn build_ast_programs(
    target_call: &Expr,
    assignments: &[(crate::expr::Ident, Expr)],
) -> Result<Vec<Program>, ParserError> {
    let mut ast_programs = Vec::new();

    if let Expr::Call { args, .. } = target_call {
        for arg in args {
            match arg {
                Expr::Identifier(name) => {
                    let mut found = false;
                    for (assign_target, assign_expr) in assignments.iter().rev() {
                        if assign_target == name {
                            if let Expr::Call { path, args: call_args } = assign_expr {
                                if path.segments.first().map(|s| s.as_str()) == Some("data") {
                                    if let Some(Expr::Literal(crate::expr::Literal::String(data_name))) = call_args.first() {
                                        ast_programs.push(Program(runtime::ast::Program::new(
                                            "data",
                                            vec![runtime::ast::Tree::Literal(crate::expr::Literal::String(data_name.clone()))]
                                        )));
                                        found = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if !found {
                        // Fallback if not a recognized data assignment
                        ast_programs.push(Program(runtime::ast::Program::new(
                            "data",
                            vec![runtime::ast::Tree::Literal(crate::expr::Literal::String("volume".to_string()))]
                        )));
                    }
                }
                Expr::Literal(lit) => {
                    ast_programs.push(Literal(lit.clone()));
                }
                _ => {
                    ast_programs.push(Program(runtime::ast::Program::new(
                        "data",
                        vec![runtime::ast::Tree::Literal(crate::expr::Literal::String("volume".to_string()))]
                    )));
                }
            }
        }
    }

    Ok(ast_programs)
}

fn parse_program(pair: pest::iterators::Pair<Rule>) -> Result<InputCode, ParserError> {
    // receives tokens, outputs Behavior and Flows
    let mut declarations = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::declaration => declarations.push(parse_declaration(inner)?),
            Rule::EOI => (),
            _ => return Err(ParserError::UnexpectedRule(inner.as_rule())),
        }
    }
    Ok(declarations)
}

fn parse_declaration(
    pair: pest::iterators::Pair<Rule>,
) -> Result<crate::behavior::InputDecl, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::import_decl => {
            let s = inner.into_inner().nth(1).unwrap().as_str();
            Ok(crate::behavior::InputDecl::Import(
                s.trim_matches('"').to_string(),
            ))
        }
        Rule::behavior_decl => parse_behavior(inner),
        Rule::flow_decl => parse_flow(inner),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_behavior(
    pair: pest::iterators::Pair<Rule>,
) -> Result<crate::behavior::InputDecl, ParserError> {
    let mut inner = pair.into_inner();
    inner.next(); // skip k_behavior
    let name = inner.next().unwrap().as_str().to_string();

    let mut inputs = Vec::new();
    let mut props_pair = None;
    let mut types_pair = None;

    for p in inner {
        match p.as_rule() {
            Rule::typed_arg_list => {
                for typed_arg in p.into_inner() {
                    let mut arg_inner = typed_arg.into_inner();
                    let arg_name = arg_inner.next().unwrap().as_str().to_string();
                    let arg_type = parse_types(arg_inner.next().unwrap())?;
                    inputs.push((arg_name, arg_type));
                }
            }
            Rule::behavior_props_block => {
                props_pair = Some(p);
            }
            Rule::types => {
                types_pair = Some(p);
            }
            _ => {}
        }
    }

    let output_type = parse_types(types_pair.unwrap())?;
    let mut bdecl = crate::behavior::BehaviorDecl {
        inputs,
        output: (name.clone(), output_type),
        operators: None,
        integers: None,
        floats: None,
        strings: None,
        weights: None,
        train: None,
        supervised_epochs: None,
    };

    if let Some(block) = props_pair {
        if let Some(props) = block.into_inner().next() {
            for prop in props.into_inner() {
                let mut prop_inner = prop.into_inner();
                let prop_name = prop_inner.next().unwrap().as_str();
                let prop_val = prop_inner.next().unwrap();

                match prop_name {
                    "weights" => bdecl.weights = Some(extract_string(&prop_val)?),
                    "train" => bdecl.train = Some(extract_bool(&prop_val)?),
                    "supervised_epochs" => {
                        bdecl.supervised_epochs = Some(extract_int(&prop_val)? as usize)
                    }
                    "operators" => bdecl.operators = Some(extract_ident_list(&prop_val)?),
                    "integers" => bdecl.integers = Some(extract_int_list(&prop_val)?),
                    "floats" => bdecl.floats = Some(extract_float_list(&prop_val)?),
                    "strings" => bdecl.strings = Some(extract_string_list(&prop_val)?),
                    _ => {
                        return Err(ParserError::SemanticError(format!(
                            "Unknown property: {}",
                            prop_name
                        )));
                    }
                }
            }
        }
    }

    Ok(crate::behavior::InputDecl::Behavior(bdecl))
}

fn parse_types(pair: pest::iterators::Pair<Rule>) -> Result<Signal, ParserError> {
    match pair.as_str() {
        "Void" => Ok(Signal::Void),
        "Float" => Ok(Signal::Float(None)),
        "Int" => Ok(Signal::Int(None)),
        "String" => Ok(Signal::String(None)),
        "Vector" => Ok(Signal::DataFrame(None)),
        "DataFrame" => Ok(Signal::DataFrame(None)),
        s => Err(ParserError::SemanticError(format!("Unknown type: {}", s))),
    }
}

fn extract_string(pair: &pest::iterators::Pair<Rule>) -> Result<String, ParserError> {
    let lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected literal".into()))?;
    let inner = lit
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected string inner".into()))?;
    Ok(inner.as_str().trim_matches('"').to_string())
}
fn extract_bool(pair: &pest::iterators::Pair<Rule>) -> Result<bool, ParserError> {
    let lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected literal".into()))?;
    let inner = lit
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected bool inner".into()))?;
    Ok(inner.as_str() == "true")
}
fn extract_int(pair: &pest::iterators::Pair<Rule>) -> Result<i64, ParserError> {
    let lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected literal".into()))?;
    let inner = lit
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected int inner".into()))?;
    inner
        .as_str()
        .parse()
        .map_err(|_| ParserError::SemanticError("Failed to parse int".into()))
}
fn extract_ident_list(
    pair: &pest::iterators::Pair<Rule>,
) -> Result<Vec<crate::expr::Ident>, ParserError> {
    let list_ident = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError(
            "Expected list identifier".into(),
        ))?;
    let mut res = Vec::new();
    for ident in list_ident.into_inner() {
        res.push(ident.as_str().to_string());
    }
    Ok(res)
}
fn extract_int_list(pair: &pest::iterators::Pair<Rule>) -> Result<Vec<i64>, ParserError> {
    let list_lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected list literal".into()))?;
    let mut res = Vec::new();
    for lit in list_lit.into_inner() {
        let inner = lit
            .into_inner()
            .next()
            .ok_or(ParserError::SemanticError("Expected int inner".into()))?;
        res.push(
            inner
                .as_str()
                .parse()
                .map_err(|_| ParserError::SemanticError("Failed to parse int".into()))?,
        );
    }
    Ok(res)
}
fn extract_float_list(pair: &pest::iterators::Pair<Rule>) -> Result<Vec<f64>, ParserError> {
    let list_lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected list literal".into()))?;
    let mut res = Vec::new();
    for lit in list_lit.into_inner() {
        let inner = lit
            .into_inner()
            .next()
            .ok_or(ParserError::SemanticError("Expected float inner".into()))?;
        res.push(
            inner
                .as_str()
                .parse()
                .map_err(|_| ParserError::SemanticError("Failed to parse float".into()))?,
        );
    }
    Ok(res)
}
fn extract_string_list(pair: &pest::iterators::Pair<Rule>) -> Result<Vec<String>, ParserError> {
    let list_lit = pair
        .clone()
        .into_inner()
        .next()
        .ok_or(ParserError::SemanticError("Expected list literal".into()))?;
    let mut res = Vec::new();
    for lit in list_lit.into_inner() {
        let inner = lit
            .into_inner()
            .next()
            .ok_or(ParserError::SemanticError("Expected string inner".into()))?;
        res.push(inner.as_str().trim_matches('"').to_string());
    }
    Ok(res)
}

fn parse_flow(
    pair: pest::iterators::Pair<Rule>,
) -> Result<crate::behavior::InputDecl, ParserError> {
    let mut inner = pair.into_inner();
    inner.next(); // k_flow
    let name = inner.next().unwrap().as_str().to_string();

    let mut body = Vec::new();
    for p in inner {
        match p.as_rule() {
            Rule::assignment_stmt => {
                let mut assn_inner = p.into_inner();
                let target = assn_inner.next().unwrap().as_str().to_string();
                let expr = parse_expr(assn_inner.next().unwrap())?;
                body.push(crate::expr::FlowStmt::Assignment {
                    target,
                    expr,
                });
            }
            Rule::expr => {
                body.push(crate::expr::FlowStmt::Expr(parse_expr(p)?));
            }
            _ => {}
        }
    }

    Ok(crate::behavior::InputDecl::Flow(
        crate::behavior::FlowDecl { name, body },
    ))
}

fn parse_expr(pair: pest::iterators::Pair<Rule>) -> Result<crate::expr::Expr, ParserError> {
    let inner = pair.clone().into_inner().next().unwrap();
    // Case 1: The expression is naturally wrapping another expression (e.g., grouped by parentheses or nested)
    if inner.as_rule() == Rule::expr {
        return parse_expr(inner);
    }

    if inner.as_rule() == Rule::arg_value {
        return parse_arg_value(inner);
    }

    // Initialize with a simple identifier first
    let ident = inner.as_str().to_string();
    let mut current_expr = crate::expr::Expr::Identifier(ident.clone());

    let mut all_inner = pair.into_inner();
    let first = all_inner.next().unwrap();

    // Case 2: The first inner pair is another expression
    if first.as_rule() == Rule::expr {
        return parse_expr(first);
    }

    // Prepare a path in case this is a function call
    let path = crate::expr::Path {
        segments: vec![first.as_str().to_string()],
    };

    // Process suffixes to determine if it's a more complex expression
    for suffix in all_inner {
        // Case 3: The expression is a function call (identifier followed by a call suffix)
        if suffix.as_rule() == Rule::call_suffix {
            let mut args = Vec::new();
            if let Some(arg_vals) = suffix.into_inner().next() {
                for val in arg_vals.into_inner() {
                    args.push(parse_expr(val)?);
                }
            }
            current_expr = crate::expr::Expr::Call {
                path: path.clone(),
                args,
            };
        }
    }

    // Fallback: If no suffixes apply, it remains a simple identifier.
    Ok(current_expr)
}

fn parse_arg_value(pair: pest::iterators::Pair<Rule>) -> Result<crate::expr::Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::literal => parse_literal(inner).map(crate::expr::Expr::Literal),
        Rule::list_literal => {
            let mut exprs = Vec::new();
            for lit in inner.into_inner() {
                exprs.push(crate::expr::Expr::Literal(parse_literal(lit)?));
            }
            Ok(crate::expr::Expr::List(exprs))
        }
        Rule::list_identifier => {
            let mut exprs = Vec::new();
            for id in inner.into_inner() {
                exprs.push(crate::expr::Expr::Identifier(id.as_str().to_string()));
            }
            Ok(crate::expr::Expr::List(exprs))
        }
        Rule::range_literal => {
            let mut lits = inner.into_inner();
            let start = crate::expr::Expr::Literal(parse_literal(lits.next().unwrap())?);
            let next_lit = parse_literal(lits.next().unwrap())?;
            let end_lit = lits.next().map(parse_literal).transpose()?;

            let (step, end) = if let Some(e) = end_lit {
                (
                    Some(Box::new(crate::expr::Expr::Literal(next_lit))),
                    Box::new(crate::expr::Expr::Literal(e)),
                )
            } else {
                (None, Box::new(crate::expr::Expr::Literal(next_lit)))
            };
            Ok(crate::expr::Expr::Range {
                start: Box::new(start),
                step,
                end,
            })
        }
        _ => Err(ParserError::SemanticError(format!(
            "Unknown arg_value: {:?}",
            inner.as_rule()
        ))),
    }
}

fn parse_literal(pair: pest::iterators::Pair<Rule>) -> Result<crate::expr::Literal, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::int_literal => Ok(crate::expr::Literal::Integer(
            inner.as_str().parse().unwrap(),
        )),
        Rule::float_literal => Ok(crate::expr::Literal::Float(inner.as_str().parse().unwrap())),
        Rule::string_literal => Ok(crate::expr::Literal::String(
            inner.as_str().trim_matches('"').to_string(),
        )),
        Rule::bool_literal => Ok(crate::expr::Literal::Boolean(inner.as_str() == "true")),
        _ => Err(ParserError::SemanticError("Unknown literal".into())),
    }
}

#[test]
fn test_parse_behavior_decl() {
    let input = r#"
        Behavior Comparator(signal: DataFrame, eps: Float, reference: DataFrame) {
            weights="behavior_1_compare.pth", train=true, supervised_epochs=100,
            operators = [add, divide, ts_mean, ts_diff, consume_float, cs_rank],
            integers = [5, 21, 252], floats = [0.1, 0.5, 0.9], strings=["volume", "adv20"]
        } -> DataFrame

        Flow volume_spike {
            volume = data("volume")
            adv20 = data("adv20")
            Comparator(volume, 0.1, adv20)
        }
    "#;
    let result = parse(input);
    println!("Success: {:?}", result.is_ok());
    if let Err(e) = &result {
        println!("Error: {}", e);
    }
    assert!(result.is_ok());
    for stmt in result.unwrap() {
        println!("{:?}", stmt);
    }
}
