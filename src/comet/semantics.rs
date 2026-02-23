use crate::comet::ast::*;
use crate::comet::symbols::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Duplicate declaration: {0}")]
    DuplicateDeclaration(String),
}

pub fn analyze(program: &Program) -> Result<SymbolTable, SemanticError> {
    let mut table = SymbolTable::new();
    
    for decl in &program.declarations {
        match decl {
            Declaration::Type(d) => {
                let info = TypeInfo {
                    name: d.name.clone(),
                    parent_constraint: d.parent_constraint.clone(),
                    properties: d.properties.clone(),
                    components: d.components.clone(),
                    structure: d.structure.clone(),
                };
                table.types.insert(d.name.clone(), info);
            },
            Declaration::Behavior(d) => {
                let info = BehaviorInfo {
                    name: d.name.clone(),
                    args: d.args.clone(),
                    return_type: d.return_type.clone(),
                };
                table.behaviors.insert(d.name.clone(), info);
            },
            Declaration::Function(d) => {
                let info = FuncInfo {
                    name: d.name.clone(),
                    params: d.params.clone(),
                    return_type: d.return_type.clone(),
                    body: d.body.clone(),
                };
                table.functions.insert(d.name.clone(), info);
            },
            Declaration::Flow(d) => {
                 let info = FlowInfo {
                     name: d.name.clone(),
                     body: d.body.clone(),
                 };
                 table.flows.insert(d.name.clone(), info);
            },
             _ => {}
        }
    }
    
    Ok(table)
}
