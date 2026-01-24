use crate::comet::ast::{Program, Declaration};
use crate::comet::symbols::{SymbolTable, TypeInfo, BehaviorInfo, ImplInfo, FuncInfo, ParamInfo, FlowInfo};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Duplicate type definition: {0}")]
    DuplicateType(String),
    #[error("Duplicate behavior definition: {0}")]
    DuplicateBehavior(String),
    #[error("Unknown parent type: {0}")]
    UnknownType(String),
    #[error("Import failed: {0}")]
    ImportError(String),
}

use std::collections::HashSet;
use std::path::PathBuf;

pub struct SemanticAnalyzer {
    pub symbol_table: SymbolTable,
    pub loaded_files: HashSet<PathBuf>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        SemanticAnalyzer {
            symbol_table: SymbolTable::new(),
            loaded_files: HashSet::new(),
        }
    }

    pub fn analyze(&mut self, program: &Program, base_path: &str) -> Result<(), SemanticError> {
        let path = PathBuf::from(base_path).canonicalize().map_err(|_| SemanticError::ImportError(format!("Invalid base path: {}", base_path)))?;
        self.loaded_files.insert(path.clone());
        self.process_program(program, &path)?;
        Ok(())
    }

    fn process_program(&mut self, program: &Program, base_path: &std::path::Path) -> Result<(), SemanticError> {
        for decl in &program.declarations {
            if let Declaration::Import(imp) = decl {
                self.load_import(&imp.path, base_path)?;
            } else {
                self.register_declaration(decl)?;
            }
        }
        Ok(())
    }

    fn load_import(&mut self, path: &str, base_path: &std::path::Path) -> Result<(), SemanticError> {
        // Construct full path
        let mut full_path = base_path.to_path_buf();
        full_path.pop(); // Remove filename
        full_path.push(path);
        
        let full_path = full_path.canonicalize()
            .map_err(|e| SemanticError::ImportError(format!("Failed to resolve path {}: {:?}", full_path.display(), e)))?;
            
        if self.loaded_files.contains(&full_path) {
            // Already loaded
            return Ok(());
        }
        self.loaded_files.insert(full_path.clone());
        
        let content = std::fs::read_to_string(&full_path)
            .map_err(|_| SemanticError::ImportError(format!("Failed to read {}", full_path.display())))?;
            
        let program = crate::comet::parser::parse(&content)
             .map_err(|e| SemanticError::ImportError(format!("Parse error in {}: {:?}", path, e)))?;
             
        self.process_program(&program, &full_path)?;
        
        Ok(())
    }

    fn register_declaration(&mut self, decl: &Declaration) -> Result<(), SemanticError> {
        match decl {
            Declaration::Type(d) => {
                if self.symbol_table.types.contains_key(&d.name) {
                    return Err(SemanticError::DuplicateType(d.name.clone()));
                }
                // TODO: specific check for "Root" or ensure parent exists (unless it's Root/Any handling)
                self.symbol_table.types.insert(d.name.clone(), TypeInfo {
                    name: d.name.clone(),
                    parent: d.parent.clone(),
                    properties: d.properties.clone(),
                    components: d.components.clone(),
                    structure: d.structure.clone(),
                });
            }
            Declaration::Behavior(d) => {
                if self.symbol_table.behaviors.contains_key(&d.name) {
                    return Err(SemanticError::DuplicateBehavior(d.name.clone()));
                }
                self.symbol_table.behaviors.insert(d.name.clone(), BehaviorInfo {
                    name: d.name.clone(),
                    args: d.args.clone(),
                    return_type: d.return_type.clone(),
                });
            }
            Declaration::Impl(d) => {
                 // We don't check for duplicate Impl names necessarily, or maybe we do?
                 // Usually Impls are anonymous or named uniquely?
                 // Comet `Impl Name("Ratio") ...` -> Identifier is "Ratio".
                 self.symbol_table.implementations.push(ImplInfo {
                    name: d.name.clone(),
                    behavior: d.behavior.clone(),
                    args: d.args.clone(),
                    constraints: d.constraints.clone(),
                    ensures: d.ensures.clone(),
                    body: d.body.stmts.clone(),
                });
            }
            Declaration::Function(d) => {
                 let params = d.params.iter().map(|p| ParamInfo {
                     name: p.name.clone(),
                     ty: p.ty.clone(),
                 }).collect();
                 self.symbol_table.functions.insert(d.name.clone(), crate::comet::symbols::FuncInfo {
                    name: d.name.clone(),
                    params: params,
                    return_type: d.return_type.clone(),
                    constraints: d.constraints.clone(),
                    ensures: d.ensures.clone(),
                });
            }
            Declaration::Flow(d) => {
                self.symbol_table.flows.insert(d.name.clone(), FlowInfo {
                    name: d.name.clone(),
                    body: d.body.clone(),
                });
            }
            Declaration::Import(_) => {
                // Imports should be handled by the loader/preprocessor before this or recursively?
                // For now ignoring or logging?
                // Ideally `analyze` should potentially load imports.
            }
            _ => {}
        }
        Ok(())
    }
}
