use crate::comet::ast::{Program, Declaration};
use crate::comet::symbols::{SymbolTable, AdtInfo, ClassInfo, InstanceInfo, FuncInfo};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SemanticError {
    #[error("Duplicate ADT definition: {0}")]
    DuplicateAdt(String),
    #[error("Duplicate type definition: {0}")]
    DuplicateType(String),
    #[error("Start function not found")]
    StartFunctionNotFound,
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
        // Handle imports? (Stub for now)
        // for import in &program.imports { ... }

        for decl in &program.declarations {
            self.register_declaration(decl)?;
        }
        Ok(())
    }

    fn register_declaration(&mut self, decl: &Declaration) -> Result<(), SemanticError> {
        match decl {
            Declaration::Adt(d) => {
                if self.symbol_table.adts.contains_key(&d.name) {
                    return Err(SemanticError::DuplicateAdt(d.name.clone()));
                }
                self.symbol_table.adts.insert(d.name.clone(), AdtInfo::from(d.clone()));
            }
            Declaration::TypeSynonym(d) => {
                // For now, treat as concrete type mapping or ignore?
                // SymbolTable doesn't have synonym map in this simplified version.
                // Ignoring for skeleton.
            }
            Declaration::Class(d) => {
                self.symbol_table.classes.insert(d.name.clone(), ClassInfo {
                    name: d.name.clone(),
                    type_vars: d.type_vars.clone(),
                    signature: d.signature.clone(),
                });
            }
            Declaration::Instance(d) => {
                self.symbol_table.instances.push(InstanceInfo {
                    class_name: d.class_name.clone(),
                    types: d.types.clone(),
                    constraints: d.constraints.clone(),
                    members: d.members.clone(),
                });
            }
            Declaration::Function(d) => {
                // If it's a top-level function logic
                self.symbol_table.functions.insert(d.name.clone(), FuncInfo::from(d.clone()));
            }
        }
        Ok(())
    }
}
