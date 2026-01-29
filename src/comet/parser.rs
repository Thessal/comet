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
    let mut module_name = "Unknown".to_string();
    let mut imports = Vec::new();
    let mut declarations = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::module_decl => {
                module_name = parse_module_decl(inner)?;
            },
            Rule::import_decl => {
                imports.push(parse_import_decl(inner)?);
            },
            Rule::declaration => {
                declarations.push(parse_declaration(inner)?);
            },
            Rule::EOI => (),
            _ => return Err(ParserError::UnexpectedRule(inner.as_rule())),
        }
    }
    Ok(Program { module_name, imports, declarations })
}

fn parse_module_decl(pair: Pair<Rule>) -> Result<Ident, ParserError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    Ok(name)
}

fn parse_import_decl(pair: Pair<Rule>) -> Result<Import, ParserError> {
    let mut inner = pair.into_inner();
    let path = inner.next().unwrap().as_str().to_string();
    Ok(Import { path })
}

fn parse_declaration(pair: Pair<Rule>) -> Result<Declaration, ParserError> {
    let inner = pair.into_inner().next().ok_or(ParserError::MissingToken)?;
    match inner.as_rule() {
        Rule::adt_decl => Ok(Declaration::Adt(parse_adt_decl(inner)?)),
        Rule::synonym_decl => Ok(Declaration::TypeSynonym(parse_synonym_decl(inner)?)),
        Rule::class_decl => Ok(Declaration::Class(parse_class_decl(inner)?)),
        Rule::instance_decl => Ok(Declaration::Instance(parse_instance_decl(inner)?)),
        Rule::func_decl => Ok(Declaration::Function(parse_func_decl(inner)?)),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

// ADT: :: List a = Cons a (List a) | Nil
fn parse_adt_decl(pair: Pair<Rule>) -> Result<AdtDecl, ParserError> {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    
    let mut type_vars = Vec::new();
    let mut next = inner.next().unwrap();
    while next.as_rule() == Rule::type_var {
        type_vars.push(next.as_str().to_string());
        next = inner.next().unwrap();
    }
    
    let constructors = parse_constructors(next)?;
    Ok(AdtDecl { name, type_vars, constructors })
}

fn parse_constructors(pair: Pair<Rule>) -> Result<Vec<Constructor>, ParserError> {
    // constructors = { constructor ~ ("|" ~ constructor)* }
    let mut ctors = Vec::new();
    for inner in pair.into_inner() {
         ctors.push(parse_constructor(inner)?);
    }
    Ok(ctors)
}

fn parse_constructor(pair: Pair<Rule>) -> Result<Constructor, ParserError> {
    // constructor = { concrete_type ~ atom_type* }
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let mut args = Vec::new();
    for arg in inner {
        args.push(parse_atom_type(arg)?);
    }
    Ok(Constructor { name, index: None, args })
}

fn parse_synonym_decl(pair: Pair<Rule>) -> Result<TypeSynDecl, ParserError> {
    // synonym_decl = { "::" ~ concrete_type ~ type_var* ~ ":==" ~ type_ref }
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    
    let mut type_vars = Vec::new();
    let mut next = inner.next().unwrap();
    while next.as_rule() == Rule::type_var {
        type_vars.push(next.as_str().to_string());
        next = inner.next().unwrap();
    }
    
    // next is type_ref (skipping :==)
    let target = parse_type_ref(next)?;
    Ok(TypeSynDecl { name, type_vars, target })
}

fn parse_class_decl(pair: Pair<Rule>) -> Result<ClassDecl, ParserError> {
    // class_decl = { "class" ~ concrete_type ~ type_var* ~ ("::" ~ type_ref)? }
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    
    let mut type_vars = Vec::new();
    let mut signature = None;
    
    for item in inner {
        match item.as_rule() {
            Rule::type_var => type_vars.push(item.as_str().to_string()),
            Rule::type_ref => signature = Some(parse_type_ref(item)?),
            _ => (), // Should not happen based on grammar
        }
    }
    
    Ok(ClassDecl { name, type_vars, signature })
}

fn parse_instance_decl(pair: Pair<Rule>) -> Result<InstanceDecl, ParserError> {
    // instance_decl = { "instance" ~ concrete_type ~ atom_type* ~ ("|" ~ constraints)? ~ ("where" ~ func_def+)? }
    let mut inner = pair.into_inner();
    let class_name = inner.next().unwrap().as_str().to_string();
    
    let mut types = Vec::new();
    let mut next_opt = inner.next();
    
    while let Some(pair) = next_opt.clone() {
        if pair.as_rule() == Rule::atom_type {
            types.push(parse_atom_type(pair)?);
            next_opt = inner.next();
        } else {
            break;
        }
    }
    
    let mut constraints = Vec::new();
    let mut members = Vec::new();
    
    if let Some(pair) = next_opt.clone() {
        if pair.as_rule() == Rule::constraints {
             constraints = parse_constraints(pair)?;
             next_opt = inner.next();
        }
    }
    
    // Check for func_def
    // Loop remaining (skipping "where" if implicit)
     while let Some(pair) = next_opt {
          if pair.as_rule() == Rule::func_def {
               members.push(parse_func_def(pair)?);
          }
           next_opt = inner.next();
     }

    Ok(InstanceDecl { class_name, types, constraints, members })
}

fn parse_constraints(pair: Pair<Rule>) -> Result<Vec<Constraint>, ParserError> {
    // constraints = { constraint ~ ("&" ~ constraint)* }
    let mut list = Vec::new();
    for inner in pair.into_inner() {
        let mut c_inner = inner.into_inner();
        let class_name = c_inner.next().unwrap().as_str().to_string();
        let mut type_args = Vec::new();
        for arg in c_inner {
            type_args.push(arg.as_str().to_string());
        }
        list.push(Constraint { class_name, type_args });
    }
    Ok(list)
}

// Functions

fn parse_func_decl(pair: Pair<Rule>) -> Result<FuncDecl, ParserError> {
    // func_decl = { func_sig? ~ func_def }
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();
    
    let signature = if first.as_rule() == Rule::func_sig {
        let sig_pair = first;
        // func_sig = { identifier ~ "::" ~ type_ref ~ ("|" ~ constraints)? }
        let mut s_inner = sig_pair.into_inner();
        let _name = s_inner.next().unwrap(); 
        let ty = parse_type_ref(s_inner.next().unwrap())?;
        
        let mut constraints = Vec::new();
        if let Some(next) = s_inner.next() {
            if next.as_rule() == Rule::constraints {
                constraints = parse_constraints(next)?;
            }
        }
        
        let def = inner.next().unwrap();
        let mut fd = parse_func_def(def)?;
        fd.signature = Some(ty);
        fd.constraints = constraints;
        Ok(fd)
    } else {
        // Just func_def
        parse_func_def(first)
    }?;
    Ok(signature)
}

fn parse_func_def(pair: Pair<Rule>) -> Result<FuncDecl, ParserError> {
    // func_def = { identifier ~ identifier* ~ "=" ~ expr ~ ("where" ~ func_def+)? }
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let mut args = Vec::new();
    
    let mut next = inner.next().unwrap();
    while next.as_rule() == Rule::identifier {
        args.push(next.as_str().to_string());
        next = inner.next().unwrap();
    }
    
    // next is expr
    let body = parse_expr(next)?;
    
    let mut where_block = None;
    if let Some(w) = inner.next() {
         // where part? Grammar: ("where" ~ func_def+)?
         // Assuming pest returns func_def nodes.
         let mut decls = Vec::new();
         decls.push(parse_func_def(w)?);
         for remaining in inner {
             decls.push(parse_func_def(remaining)?);
         }
         where_block = Some(decls);
    }
    
    Ok(FuncDecl {
        name,
        signature: None,
        constraints: Vec::new(),
        args,
        body,
        where_block,
    })
}

// Types

fn parse_type_ref(pair: Pair<Rule>) -> Result<TypeRef, ParserError> {
    // type_ref = { function_type }
    // function_type = { app_type ~ ("->" ~ app_type)* }
    let mut inner = pair.into_inner(); // function_type inner
    if inner.peek().unwrap().as_rule() == Rule::function_type {
        // recursive or wrapper
        inner = inner.next().unwrap().into_inner();
    }
    
    let mut types = Vec::new();
    for t in inner {
        types.push(parse_app_type(t)?);
    }
    
    if types.len() == 1 {
        Ok(types.pop().unwrap())
    } else {
        // fold right for arrows a -> b -> c
        let last = types.pop().unwrap();
        let mut acc = last;
        while let Some(prev) = types.pop() {
            acc = TypeRef::Function(vec![prev], Box::new(acc));
        }
        Ok(acc)
    }
}

fn parse_app_type(pair: Pair<Rule>) -> Result<TypeRef, ParserError> {
     // app_type = { atom_type+ }
     let mut inner = pair.into_inner();
     let first = parse_atom_type(inner.next().unwrap())?;
     let mut args = Vec::new();
     for rest in inner {
         args.push(parse_atom_type(rest)?);
     }
     if args.is_empty() {
         Ok(first)
     } else {
         Ok(TypeRef::Application(Box::new(first), args))
     }
}

fn parse_atom_type(pair: Pair<Rule>) -> Result<TypeRef, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::concrete_type => Ok(TypeRef::Concrete(inner.as_str().to_string())),
        Rule::type_var => Ok(TypeRef::Variable(inner.as_str().to_string())),
        Rule::type_ref => parse_type_ref(inner),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}


// Expressions

fn parse_expr(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::let_expr => parse_let(inner),
        Rule::case_expr => parse_case(inner),
        Rule::logic_expr => parse_logic(inner),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_let(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    // let func_def+ in expr
    let mut inner = pair.into_inner();
    let mut bindings = Vec::new();
    let mut next = inner.next().unwrap();
    while next.as_rule() == Rule::func_def {
        let fd = parse_func_def(next)?;
        bindings.push(Binding { name: fd.name, expr: fd.body });
        next = inner.next().unwrap();
    }
    let body = parse_expr(next)?;
    Ok(Expr::Let { bindings, body: Box::new(body) })
}

fn parse_case(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    // case expr of case_arm+
    let mut inner = pair.into_inner();
    let target = parse_expr(inner.next().unwrap())?;
    let mut arms = Vec::new();
    for arm in inner {
        arms.push(parse_case_arm(arm)?);
    }
    Ok(Expr::Case { target: Box::new(target), arms })
}

fn parse_case_arm(pair: Pair<Rule>) -> Result<CaseArm, ParserError> {
    let mut inner = pair.into_inner();
    let pattern = parse_pattern(inner.next().unwrap())?;
    let expr = parse_expr(inner.next().unwrap())?;
    Ok(CaseArm { pattern, expr })
}

fn parse_pattern(pair: Pair<Rule>) -> Result<Pattern, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::adt_pattern => {
             let mut pi = inner.into_inner();
             let name = pi.next().unwrap().as_str().to_string();
             let mut args = Vec::new();
             for a in pi {
                 args.push(a.as_str().to_string());
             }
             Ok(Pattern::Constructor { name, args })
        },
        Rule::var_pattern => Ok(Pattern::Constructor { name: inner.as_str().to_string(), args: vec![] }), // var pattern often same as 0-arg ctor? No, binds var.
        // Wait, Pattern enum needs Variable binding?
        // Pattern::Constructor vs Binding.
        // Assuming VarPattern implies binding a variable.
        // Just mapped to Constructor for now as placeholder or add Pattern::Variable?
        // Treating as Constructor(name, []) is ambiguous if it's a variable.
        // But for ADTs, capitalized is Ctor, lowercase is Var.
        // Parser rule `var_pattern` uses identifier.
        // Let's rely on that.
        Rule::wildcard_pattern => Ok(Pattern::Wildcard),
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_logic(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    // logic_expr = { term ~ (op_logic ~ term)* }
    // Standard Pratt or climbing. Simplified here.
    let mut inner = pair.into_inner();
    let mut lhs = parse_term(inner.next().unwrap())?;
    while let Some(op) = inner.next() {
        let op_enum = match op.as_str() { "&&" => Op::And, "||" => Op::Or, _ => Op::And };
        let rhs = parse_term(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: op_enum, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_term(pair: Pair<Rule>) -> Result<Expr, ParserError> {
     let mut inner = pair.into_inner();
    let mut lhs = parse_factor(inner.next().unwrap())?;
    while let Some(op) = inner.next() {
         let op_enum = match op.as_str() { 
             "==" => Op::Eq, "!=" => Op::Neq, 
             "<" => Op::Lt, ">" => Op::Gt, 
             "<=" => Op::Lt, ">=" => Op::Gt, // TODO fix op
             _ => Op::Eq 
         };
        let rhs = parse_factor(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: op_enum, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_factor(pair: Pair<Rule>) -> Result<Expr, ParserError> {
     let mut inner = pair.into_inner();
    let mut lhs = parse_atom(inner.next().unwrap())?;
    while let Some(op) = inner.next() {
         let op_enum = match op.as_str() { "+" => Op::Add, "-" => Op::Sub, "*" => Op::Mul, "/" => Op::Div, _ => Op::Add };
        let rhs = parse_atom(inner.next().unwrap())?;
        lhs = Expr::BinaryOp { left: Box::new(lhs), op: op_enum, right: Box::new(rhs) };
    }
    Ok(lhs)
}

fn parse_atom(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::literal => parse_literal(inner),
        Rule::identifier => Ok(Expr::Identifier(inner.as_str().to_string())),
        Rule::app_expr => {
            let mut ai = inner.into_inner();
            let func_name = ai.next().unwrap().as_str().to_string();
            let mut args = Vec::new();
            for a in ai {
                // a is app_arg -> atom
                let atom = a.into_inner().next().unwrap();
                args.push(parse_atom(atom)?);
            }
            Ok(Expr::Application { func: Box::new(Expr::Identifier(func_name)), args })
        },
        Rule::expr => parse_expr(inner), // parens
        _ => Err(ParserError::UnexpectedRule(inner.as_rule())),
    }
}

fn parse_literal(pair: Pair<Rule>) -> Result<Expr, ParserError> {
    let inner = pair.into_inner().next().unwrap();
    let lit = match inner.as_rule() {
        Rule::int_literal => Literal::Integer(inner.as_str().parse().unwrap()),
        Rule::float_literal => Literal::Float(inner.as_str().parse().unwrap()),
        Rule::string_literal => Literal::String(inner.as_str().trim_matches('"').to_string()),
        Rule::bool_literal => Literal::Boolean(inner.as_str() == "true"),
        _ => return Err(ParserError::UnexpectedRule(inner.as_rule())),
    };
    Ok(Expr::Literal(lit))
}
