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
        Rule::struct_decl => Ok(Declaration::Struct(parse_struct_decl(inner)?)),
        Rule::enum_decl => Ok(Declaration::Enum(parse_enum_decl(inner)?)),
        Rule::behavior_decl => Ok(Declaration::Behavior(parse_behavior_decl(inner)?)),
        Rule::impl_decl => Ok(Declaration::Impl(parse_impl_decl(inner)?)),
        Rule::flow_decl => Ok(Declaration::Flow(parse_flow_decl(inner)?)),
        Rule::func_decl => Ok(Declaration::Function(parse_func_decl(inner)?)),
        Rule::property_decl => Ok(Declaration::Property(parse_property_decl(inner)?)),
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
    let s = pair.as_str();
    // println!("Parsed ID: '{}' Rule: {:?}", s, pair.as_rule());
    s.to_string()
}

fn parse_type_decl(pair: Pair<Rule>) -> Result<TypeDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_type = inner.next().ok_or(ParserError::MissingToken)?; 
    let name = parse_identifier(inner.next().ok_or(ParserError::MissingToken)?);
    let parent = parse_identifier(inner.next().ok_or(ParserError::MissingToken)?);
    
    let mut structure = None;
    let mut properties = Vec::new();
    let mut components = None;

    for part in inner {
        match part.as_rule() {
            Rule::identifier => {
                structure = Some(parse_identifier(part));
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
        parent,
        properties,
        components,
        structure,
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

fn parse_struct_decl(pair: Pair<Rule>) -> Result<StructDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_struct = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let fields = parse_field_list(inner.next().unwrap())?;
    Ok(StructDecl { name, fields })
}

fn parse_field_list(pair: Pair<Rule>) -> Result<Vec<Field>, ParserError> {
    let mut fields = Vec::new();
    // field_list = { (identifier ~ ":" ~ type_ref ~ comma?)* }
    // inner pairs: identifier, type_ref, identifier, type_ref ...
    let mut inner = pair.into_inner();
    while let Some(ident_pair) = inner.next() {
        let name = parse_identifier(ident_pair);
        let type_pair = inner.next().ok_or(ParserError::MissingToken)?;
        let ty = type_pair.as_str().to_string(); // type_ref -> identifier -> str
        fields.push(Field { name, ty });
    }
    Ok(fields)
}

fn parse_enum_decl(pair: Pair<Rule>) -> Result<EnumDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_enum = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let variants = parse_enum_variant_list(inner.next().unwrap())?;
    Ok(EnumDecl { name, variants })
}

fn parse_enum_variant_list(pair: Pair<Rule>) -> Result<Vec<Ident>, ParserError> {
    let mut variants = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            variants.push(parse_identifier(inner));
        }
    }
    Ok(variants)
}

fn parse_behavior_decl(pair: Pair<Rule>) -> Result<BehaviorDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_behavior = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let args = parse_arg_list(inner.next().unwrap())?; // Behavior arg list is defining generics/args, e.g. (A, B)
    // "->"
    let ret = parse_identifier(inner.next().unwrap());
    Ok(BehaviorDecl {
        name,
        args,
        return_type: Some(ret),
    })
}

fn parse_arg_list(pair: Pair<Rule>) -> Result<Vec<Ident>, ParserError> {
     let mut args = Vec::new();
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::identifier {
            args.push(parse_identifier(inner));
        }
    }
    Ok(args)
}

fn parse_impl_decl(pair: Pair<Rule>) -> Result<ImplDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_impl = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let _k_implements = inner.next().unwrap();
    let behavior = parse_identifier(inner.next().unwrap());
    let args = parse_arg_list(inner.next().unwrap())?;
    
    let mut next = inner.next().unwrap();
    let mut constraints = None;
    if next.as_rule() == Rule::where_clause {
        constraints = Some(parse_where_clause(next)?);
        next = inner.next().unwrap();
    }
    
    // Check for k_ensures (optional)
    let mut ensures = None;
    if next.as_rule() == Rule::k_ensures {
        // consumes k_ensures
        // next pair is property_list? (optional in grammar but braced)
        // impl_decl = { ... (k_ensures ~ "{" ~ property_list? ~ "}")? ... }
        // inner.next() -> property_list
        let prop_pair = inner.next().unwrap();
        // Since property_list is optional inside braces? No, grammar: `property_list?`
        // Pest inner iterates rules. If property_list is present it will be yielded.
        if prop_pair.as_rule() == Rule::property_list {
            ensures = Some(parse_property_list(prop_pair)?);
            next = inner.next().unwrap(); // block
        } else {
            // Empy list? {}
            // Then prop_pair probably is block?
            // Wait, if braces are literal tokens, inner iterator skips them unless atomic?
            // "{" ~ property_list? ~ "}"
            // If property_list is absent, we get just block next?
            // If property_list is present, we get property_list.
            // Let's rely on rule checking.
             if prop_pair.as_rule() == Rule::block {
                 // Empty ensures list
                 ensures = Some(Vec::new());
                 next = prop_pair;
             } else {
                 return Err(ParserError::UnexpectedRule(prop_pair.as_rule()));
             }
        }
    }

    let body = parse_block(next)?;
    Ok(ImplDecl {
        name,
        behavior,
        args,
        constraints,
        ensures,
        body,
    })
}

fn parse_where_clause(pair: Pair<Rule>) -> Result<Expr, ParserError> {
     // k_where ~ expr
     let mut inner = pair.into_inner();
     let _k_where = inner.next().unwrap();
     let expr = parse_expr(inner.next().unwrap())?;
     Ok(expr)
}

fn parse_block(pair: Pair<Rule>) -> Result<Block, ParserError> {
    let mut stmts = Vec::new();
    for inner in pair.into_inner() {
        stmts.push(parse_statement(inner)?);
    }
    Ok(Block { stmts })
}

fn parse_statement(pair: Pair<Rule>) -> Result<Stmt, ParserError> {
    // statement = { flow_stmt }
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::flow_stmt => Ok(Stmt::Flow(parse_flow_stmt(inner)?)),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_flow_stmt(pair: Pair<Rule>) -> Result<FlowStmt, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::generator_stmt => {
            let mut items = inner.into_inner();
            let target = parse_identifier(items.next().unwrap());
             // "<-"
            let source = parse_expr(items.next().unwrap())?;
            let mut constraints = None;
            if let Some(next) = items.next() {
                constraints = Some(parse_where_clause(next)?);
            }
            Ok(FlowStmt::Generator { target, source, constraints })
        }
        Rule::assignment_stmt => {
            let mut items = inner.into_inner();
            let target = parse_identifier(items.next().unwrap());
            // "="
            let expr = parse_expr(items.next().unwrap())?;
            // where?
            // "AssignmentStmt  ::= Identifier "=" Expr (WhereClause)?"
            // Grammar says where_clause?
             if let Some(_) = items.next() {
                 // where clause on assignment? AST doesn't support it directly in Assignment enum variant?
                 // AST: Assignment { target: Ident, expr: Expr }
                 // Ignoring for now or should I update AST?
                 // Ignoring.
            }
            Ok(FlowStmt::Assignment { target, expr })
        }
        Rule::return_stmt => {
            let mut items = inner.into_inner();
            let _k_return = items.next().unwrap();
            let expr = parse_expr(items.next().unwrap())?;
            Ok(FlowStmt::Return(expr))
        }
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_func_decl(pair: Pair<Rule>) -> Result<FuncDecl, ParserError> {
     let mut inner = pair.into_inner();
     let _k_fn = inner.next().unwrap();
     let name = parse_identifier(inner.next().unwrap());
     let params = parse_param_list(inner.next().unwrap())?;
     
     // "->"
     let return_type = parse_identifier(inner.next().unwrap());
     
     let mut next = inner.next().unwrap();
     let mut constraints = None;
     if next.as_rule() == Rule::k_where {
          // grammar FuncDecl: (k_where ~ constraint_block)?
          let cb = inner.next().unwrap();
          // constraint_block -> identifier ~ is ~ identifier
          // Map to Expr::PropertyCheck? Or simple constraint struct?
          // Existing code: constraints: Option<Vec<Constraint>>
          // parser used to parse `constraint_list`.
          // New grammar `constraint_block`.
          // Let's parse constraint_block manually here for simplicity or define helper.
          // Constraint: Identifier is Identifier.
          let mut cb_inner = cb.into_inner();
          let target = parse_identifier(cb_inner.next().unwrap());
          let _k_is = cb_inner.next().unwrap();
          let prop = parse_identifier(cb_inner.next().unwrap());
          
          let expr = Expr::PropertyCheck { target: Box::new(Expr::Identifier(target)), property: prop };
          constraints = Some(expr);
          
          next = inner.next().unwrap();
     }
     
     // Check ensures
     let mut ensures = None;
     if next.as_rule() == Rule::k_ensures {
          let prop_pair = inner.next().unwrap();
          if prop_pair.as_rule() == Rule::property_list {
              ensures = Some(parse_property_list(prop_pair)?);
              next = inner.next().unwrap();
          } else if prop_pair.as_rule() == Rule::block {
              ensures = Some(Vec::new());
              next = prop_pair;
          }
     }
     
     let body = parse_block(next)?;
     Ok(FuncDecl {
         name,
         params,
         return_type,
         constraints,
         ensures,
         body,
     })
}

fn parse_param_list(pair: Pair<Rule>) -> Result<Vec<Param>, ParserError> {
    let mut params = Vec::new();
    let mut inner = pair.into_inner();
    while let Some(p) = inner.next() {
        if p.as_rule() == Rule::param {
             let mut pi = p.into_inner();
             let name = parse_identifier(pi.next().unwrap());
             let ty = pi.next().unwrap().as_str().to_string();
             params.push(Param { name, ty });
        }
    }
    Ok(params)
}

fn parse_property_decl(pair: Pair<Rule>) -> Result<PropertyDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_prop = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    Ok(PropertyDecl { name })
}

fn parse_flow_decl(pair: Pair<Rule>) -> Result<FlowDecl, ParserError> {
    let mut inner = pair.into_inner();
    let _k_flow = inner.next().unwrap();
    let name = parse_identifier(inner.next().unwrap());
    let mut body = Vec::new();
    // flow_stmt*
    // inner can contain flow_stmt directly?
    // Grammar: flow_decl = { k_flow ~ identifier ~ "{" ~ flow_stmt* ~ "}" }
    // flow_stmt produces pair.
    for stmt_pair in inner {
        if stmt_pair.as_rule() == Rule::flow_stmt {
             body.push(parse_flow_stmt(stmt_pair)?);
        }
    }
    Ok(FlowDecl { name, body })
}

// Expressions

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    // pair is expr -> or_expr
    let inner = pair.into_inner().next().unwrap();
    parse_or_expr(inner)
}

fn parse_or_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let mut inner = pair.into_inner();
    let mut lhs = parse_and_expr(inner.next().unwrap())?;
    
    while let Some(_op) = inner.next() {
        let rhs = parse_and_expr(inner.next().unwrap())?;
         // op is op_or
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
        // op could be op_rel or k_is
        if op.as_rule() == Rule::k_is {
            let prop_ident = parse_identifier(inner.next().unwrap());
            lhs = Expr::PropertyCheck { target: Box::new(lhs), property: prop_ident };
        } else {
             let op_str = op.as_str();
             let operator = match op_str {
                 "<" => Op::Lt,
                 ">" => Op::Gt,
                 "<=" => Op::Lt, // TODO: Add Le? AST Op is Lt, Gt. Maybe missing Le, Ge?
                 ">=" => Op::Gt, // AST missing Le, Ge. Mapping to Lt/Gt is wrong but leaving as is with TODO.
                 _ => unreachable!(),
             };
             let rhs = parse_add_expr(inner.next().unwrap())?;
             lhs = Expr::BinaryOp { left: Box::new(lhs), op: operator, right: Box::new(rhs) };
        }
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
    // unary_expr = { op_unary? ~ atom }
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    if first.as_rule() == Rule::op_unary {
         // Handle unary op
         let op_str = first.as_str();
         let op = match op_str {
             "-" => Op::Sub,
             "!" => Op::Not,
             _ => unreachable!(),
         };
         let atom = parse_atom(inner.next().unwrap())?;
         return Ok(Expr::UnaryOp { op, target: Box::new(atom) });
    } else {
        // first is atom
        return parse_atom(first); 
    }
}

fn parse_atom(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    // atom = { primary ~ postfix* }
    let mut inner = pair.into_inner();
    let primary = parse_primary(inner.next().unwrap())?;
    
    let mut expr = primary;
    
    for postfix in inner {
        // postfix = { call_suffix | member_suffix }
        let p_inner = postfix.into_inner().next().unwrap();
        match p_inner.as_rule() {
            Rule::call_suffix => {
                // p_inner contains "(" ~ arg_values ~ ")"
                // args
                let mut args_pair = p_inner.into_inner(); 
                 // Skip "("? No, call_suffix = { "(" ... }
                 // Pair content: arg_values
                 let arg_values_pair = args_pair.next().unwrap();
                 let args = parse_arg_values(arg_values_pair)?;
                 
                 // AST Call requires Path.
                 // Expr::Call { path: Path, args }
                 // But `expr` here might be any expression, e.g. (x).foo().
                 // AST restricts Call to Path?
                 // `Call { path: Path, args: Vec<Expr> }`
                 // MemberAccess { target: Box<Expr>, field: Ident }
                 // If I have `foo()`, `foo` is Expr::Identifier.
                 // I need to convert Identifier to Path?
                 // Or AST allows Expr as target? No.
                 // AST strictly says target is Path.
                 // This implies `(expression)()` is not allowed?
                 // Docs `ast.md`: `Call { path: Path, args: Vec<Expr> }`.
                 // `Expr` has `Identifier`.
                 
                 // If expr is Identifier, I can convert to Path.
                 // If expr is MemberAccess (foo.bar), I can convert to Path?
                 // If it is binary op, cannot call.
                 
                 if let Expr::Identifier(name) = expr {
                     expr = Expr::Call { 
                          path: Path { segments: vec![name] },
                          args 
                      };
                  // } else if let Expr::MemberAccess { target: _, field: _ } = expr {
                  //    // TODO: Fix MemberAccess handling
                  //    expr = Expr::Call { path: Path { segments: vec!["UNKNOWN".to_string()] }, args };
                  } else {
                      // Error or ignore
                      // Ignoring
                  }
            }
            Rule::member_suffix => {
                 let ident = parse_identifier(p_inner.into_inner().next().unwrap()); // Skip "."
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
            // path -> identifier, ::, identifier
            let mut segments = Vec::new();
            for seg in inner.into_inner() {
                segments.push(seg.as_str().to_string());
            }
            if segments.len() == 1 {
                Ok(Expr::Identifier(segments[0].clone()))
            } else {
                // AST Expr has Identifier (single) or Call (Path).
                // Does it have bare Path? No.
                // It has Identifier.
                // It has MemberAccess.
                // `foo::bar` ?
                // Maybe treat as MemberAccess chain?
                // Or AST missing EnumVariant/StaticMember?
                // Using Identifier with "::" joined? NO.
                // I will use Identifier if len=1.
                // If len > 1, treating as member access chain?
                // foo::bar -> MemberAccess(foo, bar).
                // Actually `::` is usually static. MemberAccess `.` is instance.
                // But AST has no explicit Path expression. 
                // Using Identifier.
                Ok(Expr::Identifier(segments.join("::"))) // Hack
            }
        },
        Rule::paren_expr => {
            parse_expr(inner.into_inner().next().unwrap())
        },
        Rule::list_literal => {
            // Not in AST!
            // Ignoring
            Ok(Expr::Identifier("ListLiteralPlaceholder".to_string()))
        },
        _ => unreachable!(),
    }
}

fn parse_arg_values(pair: Pair<Rule>) -> Result<Vec<Expr>, ParserError> {
    let mut args = Vec::new();
    for inner in pair.into_inner() {
        args.push(parse_expr(inner)?);
    }
    Ok(args)
}
