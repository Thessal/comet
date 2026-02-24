mod comet;

use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: comet <file.cm>");
        return;
    }
    
    let filename = &args[1];
    let content = fs::read_to_string(filename).expect("Failed to read file");
    
    match comet::parser::parse(&content) {
        Ok(mut program) => {
             println!("Parsed successfully!");
             println!("--- AST Representation ---");
             println!("{}", program);
             println!("--------------------------");
             println!("Total declarations: {}", program.declarations.len());
             for d in &program.declarations {
                 let t = match d {
                     comet::ast::Declaration::Import(_) => "Import",
                     comet::ast::Declaration::Behavior(_) => "Behavior",
                     comet::ast::Declaration::Function(_) => "Function",
                     comet::ast::Declaration::Flow(_) => "Flow",
                 };
                 println!("Decl: {}", t);
             }
             // Handle simple imports
             let mut all_decls = Vec::new();
             for decl in &program.declarations {
                 if let comet::ast::Declaration::Import(imp) = decl {
                     let mut import_path = std::path::PathBuf::from(filename);
                     import_path.pop();
                     let import_str = imp.path.replace("\"", "").replace(" ", "");
                     import_path.push(import_str);
                     
                     if let Ok(imp_content) = fs::read_to_string(&import_path) {
                         match comet::parser::parse(&imp_content) {
                             Ok(imp_prog) => all_decls.extend(imp_prog.declarations),
                             Err(e) => println!("Error parsing {:?}: {:?}", import_path, e),
                         }
                     } else {
                         println!("Error reading file: {:?}", import_path);
                     }
                 } else {
                     all_decls.push(decl.clone());
                 }
             }
             program.declarations = all_decls;
             
             let mut library = Vec::new();
             let mut behaviors = std::collections::HashMap::new();
             for decl in &program.declarations {
                 match decl {
                     comet::ast::Declaration::Function(f) => {
                         let mut args = Vec::new();
                         for p in &f.params {
                             args.push((p.name.clone(), p.constraint.clone()));
                         }
                         library.push(comet::synthesis::FnSignature {
                             name: f.name.clone(),
                             args,
                             return_constraint: f.return_constraint.clone(),
                         });
                     },
                     comet::ast::Declaration::Behavior(b) => {
                         behaviors.insert(b.name.clone(), b.clone());
                     },
                     _ => {}
                 }
             }

             println!("Evaluating flows directly since semantics module is removed for redesign:");
             for decl in &program.declarations {
                 if let comet::ast::Declaration::Flow(flow) = decl {
                     println!("Synthesizing flow: {}", flow.name);
                     println!("--- AST Flow Body ---");
                     for stmt in &flow.body {
                         println!("{:#?}", stmt);
                     }
                     println!("---------------------");

                     let mut env_map = std::collections::HashMap::new();
                     for stmt in &flow.body {
                         if let comet::ast::FlowStmt::Assignment { target, expr } = stmt {
                             env_map.insert(target.clone(), expr.clone());
                         }
                     }

                     for stmt in &flow.body {
                         if let comet::ast::FlowStmt::Expr(expr) = stmt {
                             let substituted_expr = comet::synthesis::substitute_expr(expr, &env_map);
                             println!("Synthesizing fully substituted target expression: {:#?}", substituted_expr);
                             match comet::synthesis::Synthesizer::synthesize_expr(&substituted_expr, &behaviors, &library) {
                                 Ok(real_exprs) => {
                                     
                                     let mut graphs = Vec::new();
                                     for (i, real_forest) in real_exprs.iter().enumerate() {
                                         println!("--- Context {} ---", i);
                                         for (j, tree) in real_forest.iter().enumerate() {
                                             println!("AST equivalent [Tree {}]: {:?}", j, tree);
                                         }
                                         let g = comet::ir::ExecutionGraph::from_forest(real_forest);
                                         graphs.push(g);
                                     }
                                     
                                     let inkwell_ctx = inkwell::context::Context::create();
                                     let codegen = comet::codegen::Codegen::new(&inkwell_ctx, &flow.name);
                                     
                                     let ir_string = codegen.generate_ir(&graphs);
                                     println!("Generated LLVM IR for {}:\n{}", flow.name, ir_string);
                                     
                                     match codegen.emit_library(&flow.name) {
                                          Ok(_) => println!("Successfully compiled {}.so library!", flow.name),
                                          Err(e) => eprintln!("Failed to compile library: {}", e),
                                     }
                                 },
                                 Err(e) => eprintln!("Synthesis error: {}", e),
                             }
                         }
                     }
                 }
             }
        },
        Err(e) => {
             eprintln!("Parse error: {:?}", e);
        }
    }
}
