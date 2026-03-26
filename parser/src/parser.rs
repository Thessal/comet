use crate::program::*;
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use thiserror::Error;

use std::collections::HashSet;

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

pub fn parse(input: &str) -> Result<Program, ParserError> {
    let mut pairs = CometParser::parse(Rule::program, input)?;
    let program_pair = pairs.next().ok_or(ParserError::MissingToken)?;
    Ok(parse_program(program_pair)?)
}

fn parse_program(pair: Pair<Rule>) -> Result<Program, ParserError> {
    let mut declarations = Vec::new();
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::declaration => declarations.push(parse_declaration(inner)?),
            Rule::EOI => (),
            _ => return Err(ParserError::UnexpectedRule(inner.as_rule())),
        }
    }

    let prog = Program { declarations };
    validate_program(&prog)?;
    Ok(prog)
}

fn validate_program(program: &Program) -> Result<(), ParserError> {
    let mut known_functions = HashSet::new();
    for decl in &program.declarations {
        match decl {
            Declaration::Behavior(b) => {
                known_functions.insert(b.name.clone());
            }
            _ => {}
        }
    }

    for decl in &program.declarations {
        if let Declaration::Flow(flow) = decl {
            for stmt in &flow.body {
                if let FlowStmt::Expr(expr) = stmt {
                    validate_expr(expr, &known_functions)?;
                } else if let FlowStmt::Assignment { expr, .. } = stmt {
                    validate_expr(expr, &known_functions)?;
                }
            }
        }
    }

    Ok(())
}

fn validate_expr(expr: &Expr, known_functions: &HashSet<String>) -> Result<(), ParserError> {
    match expr {
        Expr::Call { path, args } => {
            let _func_name = path.segments.last().unwrap();
            // In the future, this should check against an injected stdlib registry.
            // For now, we allow any function call structure.
            for arg in args {
                validate_expr(arg, known_functions)?;
            }
            Ok(())
        }

        Expr::MemberAccess { target, .. } => validate_expr(target, known_functions),
        Expr::List(exprs) => {
            for e in exprs {
                validate_expr(e, known_functions)?;
            }
            Ok(())
        }
        Expr::Range { start, step, end } => {
            validate_expr(start, known_functions)?;
            if let Some(s) = step {
                validate_expr(s, known_functions)?;
            }
            validate_expr(end, known_functions)?;
            Ok(())
        }
        _ => Ok(()),
    }
}

fn parse_declaration(pair: Pair<Rule>) -> Result<Declaration, ParserError> {
    let inner = pair.into_inner().next().ok_or(ParserError::MissingToken)?;
    match inner.as_rule() {
        Rule::import_decl => Ok(Declaration::Import(parse_import_decl(inner)?)),
        Rule::behavior_decl => Ok(Declaration::Behavior(parse_behavior_decl(inner)?)),
        Rule::flow_decl => Ok(Declaration::Flow(parse_flow_decl(inner)?)),

        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_import_decl(pair: Pair<Rule>) -> Result<ImportDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_import = inner.next().unwrap();
    let lit_pair = inner.next().unwrap();
    let path = lit_pair.as_str().trim_matches('"').to_string();
    Ok(ImportDecl { path })
}

fn parse_identifier(pair: Pair<Rule>) -> Ident {
    pair.as_str().to_string()
}

// Constraints Parsing

fn parse_type_name(pair: Pair<Rule>) -> Result<TypeDecl, ParserError> {
    match pair.as_str() {
        "DataFrame" => Ok(TypeDecl::DataFrame),
        "Matrix" => Ok(TypeDecl::Matrix),
        "Vector" => Ok(TypeDecl::Vector),
        "String" => Ok(TypeDecl::String),
        "Float" => Ok(TypeDecl::Float),
        "Bool" => Ok(TypeDecl::Bool),
        "Void" => Ok(TypeDecl::Void),
        _ => Err(ParserError::UnexpectedRule(pair.as_rule())),
    }
}

// Declaration Parsing
fn parse_typed_arg_list(pair: Pair<Rule>) -> Result<Vec<TypedArg>, ParserError> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::typed_arg {
            let mut i = inner.into_inner();
            let name = parse_identifier(i.next().unwrap());
            let type_decl = parse_type_name(i.next().unwrap())?;
            args.push(TypedArg { name, type_decl });
        }
    }
    Ok(args)
}

fn parse_behavior_decl(pair: Pair<Rule>) -> Result<BehaviorDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_behavior = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let args = parse_typed_arg_list(inner.next().unwrap())?;

    let mut weights = None;
    let mut train = None;
    let mut supervised_samples = None;

    let mut next_pair = inner.next().unwrap();

    if next_pair.as_rule() == Rule::behavior_props_block {
        if let Some(props) = next_pair.into_inner().next() {
            // props is behavior_props
            for prop in props.into_inner() {
                let mut prop_inner = prop.into_inner();
                let key = parse_identifier(prop_inner.next().unwrap());
                let val_pair = prop_inner.next().unwrap(); // Rule::literal
                let lit_inner = val_pair.into_inner().next().unwrap();

                match key.as_str() {
                    "weights" => {
                        if lit_inner.as_rule() == Rule::string_literal {
                            weights = Some(lit_inner.as_str().trim_matches('"').to_string());
                        } else {
                            return Err(ParserError::SemanticError(
                                "weights must be a string".into(),
                            ));
                        }
                    }
                    "train" => {
                        if lit_inner.as_rule() == Rule::bool_literal {
                            train = Some(lit_inner.as_str() == "true");
                        } else {
                            return Err(ParserError::SemanticError(
                                "train must be a boolean".into(),
                            ));
                        }
                    }
                    "supervised_samples" => {
                        if lit_inner.as_rule() == Rule::int_literal {
                            supervised_samples =
                                Some(lit_inner.as_str().parse().map_err(|_| {
                                    ParserError::SemanticError("Invalid supervised_samples".into())
                                })?);
                        } else {
                            return Err(ParserError::SemanticError(
                                "supervised_samples must be an integer".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(ParserError::SemanticError(format!(
                            "Unknown behavior property: {}",
                            key
                        )));
                    }
                }
            }
        }
        next_pair = inner.next().unwrap(); // advance to constraint
    }

    let return_type = parse_type_name(next_pair)?;

    Ok(BehaviorDecl {
        name,
        args,
        return_type,
        weights,
        train,
        supervised_samples,
    })
}

fn parse_flow_decl(pair: Pair<Rule>) -> Result<FlowDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_flow = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let block = parse_block(inner.next().unwrap())?;

    let mut body = Vec::new();
    for stmt in block.stmts {
        match stmt {
            Stmt::Flow(f) => body.push(f),
            Stmt::Expr(e) => body.push(FlowStmt::Expr(e)), // Return
        }
    }

    Ok(FlowDecl { name, body })
}

fn parse_block(pair: Pair<Rule>) -> Result<Block, ParserError> {
    let mut stmts = Vec::new();
    for inner in pair.into_inner() {
        stmts.push(parse_statement(inner)?);
    }
    Ok(Block { stmts })
}

fn parse_statement(pair: Pair<Rule>) -> Result<Stmt, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::assignment_stmt => {
            let mut i = inner.into_inner();
            let target = parse_identifier(i.next().unwrap());
            let expr = parse_expr(i.next().unwrap())?;
            Ok(Stmt::Flow(FlowStmt::Assignment { target, expr }))
        }
        Rule::expr_stmt => {
            let expr = parse_expr(inner.into_inner().next().unwrap())?;
            Ok(Stmt::Expr(expr))
        }
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

// Expressions

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    parse_atom(inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path as StdPath;

    #[test]
    fn test_generate_ast_dumps() {
        // Find the examples directory relative to the parser crate workspace
        let examples_dir = StdPath::new("../examples");
        assert!(
            examples_dir.exists(),
            "Examples directory not found at ../examples"
        );

        for entry in fs::read_dir(examples_dir).expect("Failed to read examples directory") {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("cm") {
                let content = fs::read_to_string(&path).expect("Failed to read .cm file");
                println!("--- Parsing file: {:?} ---", path);

                match parse(&content) {
                    Ok(ast) => {
                        let ast_dump = format!("{:#?}", ast);
                        let mut rs_path = path.clone();
                        rs_path.set_extension("rs");

                        fs::write(&rs_path, ast_dump).unwrap_or_else(|e| {
                            panic!("Failed to write AST to {:?}: {}", rs_path, e)
                        });

                        println!("AST successfully dumped to {:?}", rs_path);
                    }
                    Err(e) => {
                        println!("Error parsing {:?}:\n{:?}", path, e);
                        panic!("Failed to parse example {:?}", path);
                    }
                }
            }
        }
    }
}

fn parse_atom(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let mut inner = pair.into_inner();
    let primary = parse_primary(inner.next().unwrap())?;

    let mut expr = primary;

    for postfix in inner {
        let p_inner = postfix.into_inner().next().unwrap();
        match p_inner.as_rule() {
            Rule::call_suffix => {
                let mut args_pair = p_inner.into_inner();
                let arg_values_pair = args_pair.next().unwrap();
                let args = parse_arg_values(arg_values_pair)?;

                if let Expr::Identifier(name) = expr {
                    expr = Expr::Call {
                        path: Path {
                            segments: vec![name],
                        },
                        args,
                    };
                } else {
                    // Handle other cases if needed
                }
            }
            Rule::member_suffix => {
                let ident = parse_identifier(p_inner.into_inner().next().unwrap());
                expr = Expr::MemberAccess {
                    target: Box::new(expr),
                    field: ident,
                };
            }
            _ => unreachable!(),
        }
    }
    Ok(expr)
}

fn parse_primary(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::literal => {
            let lit_inner = inner.into_inner().next().unwrap();
            match lit_inner.as_rule() {
                Rule::int_literal => Ok(Expr::Literal(Literal::Integer(
                    lit_inner.as_str().parse().unwrap(),
                ))),
                Rule::float_literal => Ok(Expr::Literal(Literal::Float(
                    lit_inner.as_str().parse().unwrap(),
                ))),
                Rule::string_literal => Ok(Expr::Literal(Literal::String(
                    lit_inner.as_str().trim_matches('"').to_string(),
                ))),
                Rule::bool_literal => Ok(Expr::Literal(Literal::Boolean(
                    lit_inner.as_str() == "true",
                ))),
                _ => unreachable!(),
            }
        }
        Rule::path => {
            let mut segments = Vec::new();
            for seg in inner.into_inner() {
                segments.push(seg.as_str().to_string());
            }
            if segments.len() == 1 {
                Ok(Expr::Identifier(segments[0].clone()))
            } else {
                Ok(Expr::Identifier(segments.join("::")))
            }
        }
        Rule::paren_expr => parse_expr(inner.into_inner().next().unwrap()),
        Rule::list_literal => {
            let mut exprs = Vec::new();
            for e in inner.into_inner() {
                exprs.push(parse_expr(e)?);
            }
            Ok(Expr::List(exprs))
        }
        Rule::range_literal => {
            let mut exprs = Vec::new();
            for e in inner.into_inner() {
                exprs.push(parse_expr(e)?);
            }
            if exprs.len() == 2 {
                let end = exprs.pop().unwrap();
                let start = exprs.pop().unwrap();
                Ok(Expr::Range {
                    start: Box::new(start),
                    step: None,
                    end: Box::new(end),
                })
            } else if exprs.len() == 3 {
                let end = exprs.pop().unwrap();
                let step = exprs.pop().unwrap();
                let start = exprs.pop().unwrap();
                Ok(Expr::Range {
                    start: Box::new(start),
                    step: Some(Box::new(step)),
                    end: Box::new(end),
                })
            } else {
                unreachable!()
            }
        }
        _ => unreachable!(),
    }
}

fn parse_arg_values(pair: Pair<Rule>) -> Result<Vec<Expr>, ParserError> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::arg_value {
            args.push(parse_expr(inner.into_inner().next().unwrap())?);
        }
    }
    Ok(args)
}
