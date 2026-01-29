use pest::Parser;
use pest_derive::Parser;
use crate::comet::ast::*;
use pest::iterators::Pair;
use thiserror::Error;

#[derive(Parser)]
#[grammar = "comet/grammar.pest"]
pub struct CometParser;

#[derive(Error, Debug)]
pub enum ParserError {
    #[error("Pest error: {0}")]
    PestError(#[from] pest::error::Error<Rule>),
    #[error("Unexpected rule: {0:?}")]
    UnexpectedRule(Rule),
    #[error("Missing token")]
    MissingToken,
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
    Ok(Program { declarations })
}

fn parse_declaration(pair: Pair<Rule>) -> Result<Declaration, ParserError> {
    let inner = pair.into_inner().next().ok_or(ParserError::MissingToken)?;
    match inner.as_rule() {
        Rule::import_decl => Ok(Declaration::Import(parse_import_decl(inner)?)),
        Rule::type_decl => Ok(Declaration::Type(parse_type_decl(inner)?)),
        Rule::behavior_decl => Ok(Declaration::Behavior(parse_behavior_decl(inner)?)),
        Rule::flow_decl => Ok(Declaration::Flow(parse_flow_decl(inner)?)),
        Rule::func_decl => Ok(Declaration::Function(parse_func_decl(inner)?)),
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

fn parse_constraint(pair: Pair<Rule>) -> Result<Constraint, ParserError> {
    // constraint -> union_constraint
    let inner = pair.into_inner().next().unwrap();
    parse_union_constraint(inner)
}

fn parse_union_constraint(pair: Pair<Rule>) -> Result<Constraint, ParserError> {
    let mut inner = pair.into_inner();
    let first = parse_add_constraint(inner.next().unwrap())?;
    
    let mut constraints = vec![first];
    while let Some(next_pair) = inner.next() {
        // Skipped "|" token implicitly? Pest usually handles literals if they are separate tokens.
        // In grammar: union_constraint = { add_constraint ~ ("|" ~ add_constraint)* }
        // The "|" is atomic/literal, not a rule unless named.
        // Pest Pairs skipping unassigned literals? Yes usually.
        constraints.push(parse_add_constraint(next_pair)?);
    }
    
    if constraints.len() == 1 {
        Ok(constraints.pop().unwrap())
    } else {
        Ok(Constraint::Union(constraints))
    }
}

fn parse_add_constraint(pair: Pair<Rule>) -> Result<Constraint, ParserError> {
    // add_constraint = { sub_constraint ~ (sub_constraint)* }
    let mut inner = pair.into_inner();
    let first = parse_sub_constraint(inner.next().unwrap())?;
    
    let mut constraints = vec![first];
    while let Some(next_pair) = inner.next() {
        constraints.push(parse_sub_constraint(next_pair)?);
    }
    
    if constraints.len() == 1 {
        Ok(constraints.pop().unwrap())
    } else {
        Ok(Constraint::Addition(constraints))
    }
}

fn parse_sub_constraint(pair: Pair<Rule>) -> Result<Constraint, ParserError> {
    // sub_constraint = { atom_constraint ~ ("-" ~ atom_constraint)? }
    let mut inner = pair.into_inner();
    let lhs = parse_atom_constraint(inner.next().unwrap())?;
    
    if let Some(rhs_pair) = inner.next() {
        let rhs = parse_atom_constraint(rhs_pair)?;
        Ok(Constraint::Subtraction(Box::new(lhs), Box::new(rhs)))
    } else {
        Ok(lhs)
    }
}

fn parse_atom_constraint(pair: Pair<Rule>) -> Result<Constraint, ParserError> {
    // atom_constraint = { ("(" ~ constraint ~ ")") | identifier | string_literal }
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::constraint => parse_constraint(inner),
        Rule::identifier => Ok(Constraint::Atom(parse_identifier(inner))),
        Rule::string_literal => Ok(Constraint::Atom(inner.as_str().trim_matches('"').to_string())),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

// Declaration Parsing

fn parse_type_decl(pair: Pair<Rule>) -> Result<TypeDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_type = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    
    let mut parent_constraint = None;
    let mut properties = Vec::new();
    let mut components = None;
    
    // Iterate remaining
    while let Some(part) = inner.next() {
        match part.as_rule() {
            Rule::constraint => {
                parent_constraint = Some(parse_constraint(part)?);
            },
            Rule::property_list => {
                properties = parse_property_list(part)?;
            },
            Rule::component_list => {
                components = Some(parse_component_list(part)?);
            },
            _ => {}
        }
    }

    Ok(TypeDecl {
        name,
        parent_constraint,
        properties,
        components,
        structure: None, // Deprecated/Removed from grammar? Grammar doesn't have structure slot anymore in new version
    })
}

fn parse_component_list(pair: Pair<Rule>) -> Result<Vec<Ident>, ParserError> {
    let mut comps = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            comps.push(parse_identifier(inner));
        }
    }
    Ok(comps)
}

fn parse_property_list(pair: Pair<Rule>) -> Result<Vec<Ident>, ParserError> {
    let mut props = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            props.push(parse_identifier(inner));
        }
    }
    Ok(props)
}

fn parse_typed_arg_list(pair: Pair<Rule>) -> Result<Vec<TypedArg>, ParserError> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::typed_arg {
            let mut i = inner.into_inner();
            let name = parse_identifier(i.next().unwrap());
            let constraint = parse_constraint(i.next().unwrap())?;
            args.push(TypedArg { name, constraint });
        }
    }
    Ok(args)
}

fn parse_behavior_decl(pair: Pair<Rule>) -> Result<BehaviorDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_behavior = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let args = parse_typed_arg_list(inner.next().unwrap())?;
    let return_type = parse_constraint(inner.next().unwrap())?;
    
    Ok(BehaviorDecl {
        name,
        args,
        return_type,
    })
}

fn parse_func_decl(pair: Pair<Rule>) -> Result<FuncDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_fn = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    
    let mut params = Vec::new();
    if let Some(p) = inner.peek() {
        if p.as_rule() == Rule::typed_arg_list {
            params = parse_typed_arg_list(inner.next().unwrap())?;
        }
    }
    
    let return_type = parse_constraint(inner.next().unwrap())?;
    
    // where clause removed

    // ensures removed
    
    let body = parse_block(inner.next().unwrap())?;
    
    Ok(FuncDecl {
        name,
        params,
        return_type,
        body,
    })
}

fn parse_flow_decl(pair: Pair<Rule>) -> Result<FlowDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_flow = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let expr = parse_expr(inner.next().unwrap())?;
    
    // Flow body is just one assignment "name = expr" effectively in new grammar
    // grammar: flow_decl = { k_flow ~ identifier ~ "=" ~ expr }
    // AST: FlowDecl { name, body: Vec<FlowStmt> }
    // We convert this to Single Assignment Stmt
    
    let stmt = FlowStmt::Assignment { target: "result".to_string(), expr };
    
    Ok(FlowDecl { name, body: vec![stmt] })
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
        },
        Rule::return_stmt => {
            let mut i = inner.into_inner();
            let _k_return = i.next().unwrap();
            let expr = parse_expr(i.next().unwrap())?;
            Ok(Stmt::Return(expr))
        },
        Rule::expr_stmt => {
            let expr = parse_expr(inner.into_inner().next().unwrap())?;
            Ok(Stmt::Expr(expr))
        },
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}


// Expressions

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    parse_or_expr(inner)
}

fn parse_or_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_and_expr(inner.next().unwrap())?;
    
    while let Some(_op) = inner.next() {
        let rhs = parse_and_expr(inner.next().unwrap())?;
         lhs = Expr::BinaryOp { left: Box::new(lhs), op: Op::Or, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_and_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
     let mut inner = pair.into_inner();
    let mut lhs = parse_eq_expr(inner.next().unwrap())?;
    
    while let Some(_op) = inner.next() {
        let rhs = parse_eq_expr(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: Op::And, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_eq_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
      let mut inner = pair.into_inner();
    let mut lhs = parse_rel_expr(inner.next().unwrap())?;
    
    while let Some(op) = inner.next() {
        let op_str = op.as_str();
        let operator = match op_str {
            "==" => Op::Eq,
            "!=" => Op::Neq,
            _ => unreachable!(),
        };
        let rhs = parse_rel_expr(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: operator, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_rel_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_add_expr(inner.next().unwrap())?;
    
    while let Some(op) = inner.next() {
         let op_str = op.as_str();
         let operator = match op_str {
             "<" => Op::Lt,
             ">" => Op::Gt,
             "<=" => Op::Lt, 
             ">=" => Op::Gt, 
             _ => unreachable!(),
         };
         let rhs = parse_add_expr(inner.next().unwrap())?;
         lhs = Expr::BinaryOp { left: Box::new(lhs), op: operator, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_add_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
     let mut inner = pair.into_inner();
    let mut lhs = parse_mul_expr(inner.next().unwrap())?;
    while let Some(op) = inner.next() {
        let op_str = op.as_str();
        let operator = match op_str {
            "+" => Op::Add,
            "-" => Op::Sub,
            _ => unreachable!(),
        };
        let rhs = parse_mul_expr(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: operator, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_mul_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
      let mut inner = pair.into_inner();
    let mut lhs = parse_unary_expr(inner.next().unwrap())?;
    while let Some(op) = inner.next() {
        let op_str = op.as_str();
        let operator = match op_str {
            "*" => Op::Mul,
            "/" => Op::Div,
            _ => unreachable!(),
        };
        let rhs = parse_unary_expr(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: operator, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_unary_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    if first.as_rule() == Rule::op_unary {
         let op_str = first.as_str();
         let op = match op_str {
             "-" => Op::Sub,
             "!" => Op::Not,
             _ => unreachable!(),
         };
         let atom = parse_atom(inner.next().unwrap())?;
         return Ok(Expr::UnaryOp { op, target: Box::new(atom) });
    } else {
        return parse_atom(first); 
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
                          path: Path { segments: vec![name] },
                          args 
                      };
                  } else {
                      // Handle other cases if needed
                  }
            }
            Rule::member_suffix => {
                 let ident = parse_identifier(p_inner.into_inner().next().unwrap());
                 expr = Expr::MemberAccess { target: Box::new(expr), field: ident };
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
                 Rule::int_literal => Ok(Expr::Literal(Literal::Integer(lit_inner.as_str().parse().unwrap()))),
                 Rule::float_literal => Ok(Expr::Literal(Literal::Float(lit_inner.as_str().parse().unwrap()))),
                 Rule::string_literal => Ok(Expr::Literal(Literal::String(lit_inner.as_str().trim_matches('"').to_string()))),
                 Rule::bool_literal => Ok(Expr::Literal(Literal::Boolean(lit_inner.as_str() == "true"))),
                 _ => unreachable!(),
             }
        },
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
        },
        Rule::paren_expr => {
            parse_expr(inner.into_inner().next().unwrap())
        },
        Rule::list_literal => {
            Ok(Expr::Identifier("ListLiteralPlaceholder".to_string()))
        },
        _ => unreachable!(),
    }
}

fn parse_arg_values(pair: Pair<Rule>) -> Result<Vec<ArgValue>, ParserError> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::arg_value {
            let mut arg_inner = inner.into_inner();
            let first = arg_inner.next().unwrap();
            
            let (name, expr) = if first.as_rule() == Rule::identifier {
                // named arg: name = expr
                let name = parse_identifier(first);
                let expr = parse_expr(arg_inner.next().unwrap())?;
                (Some(name), expr)
            } else {
                // positional arg: expr
                // parse_expr wants a pair, specifically or_expr which is what parse_expr expects.
                // first IS an expr rule (or_expr et al) ? No, grammar says:
                // arg_value = { (identifier ~ "=")? ~ expr }
                // so first is likely 'expr' rule if no identifier.
                // Pest might skip expr rule if atomic? no.
                // first is the expr pair.
                let expr = parse_expr(first)?;
                (None, expr)
            };
            args.push(ArgValue { name, value: expr });
        }
    }
    Ok(args)
}
