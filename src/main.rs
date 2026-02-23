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
                     comet::ast::Declaration::Type(_) => "Type",
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
             
             // let mut analyzer = comet::semantics::SemanticAnalyzer::new();
             // match analyzer.analyze(&program, filename) {
             /*
             match comet::semantics::analyze(&program) {
                 Ok(symbol_table) => {
                     println!("Semantic analysis passed!");
                     println!("Symbol Table Stats:");
                     println!("Types: {}", symbol_table.types.len());
                     println!("Behaviors: {}", symbol_table.behaviors.len());
                     println!("Functions: {}", symbol_table.functions.len());
                     println!("Flows: {}", symbol_table.flows.len());
                     
                     // Synthesis Step
                     let synthesizer = comet::synthesis::Synthesizer::new(&symbol_table);
                     for (flow_name, _) in &symbol_table.flows {
                        println!("Synthesizing flow: {}", flow_name);
                         match synthesizer.synthesize(flow_name) {
                             Ok(contexts) => {
                                 println!("Synthesis successful for {}! Count: {}", flow_name, contexts.len());
                                 
                                 // Initialize the LLVM codegen context
                                 let inkwell_ctx = inkwell::context::Context::create();
                                 let codegen = comet::codegen::Codegen::new(&inkwell_ctx, flow_name);
                                 
                                 // Generate the internal LLVM IR structure
                                 let _ir_string = codegen.generate_ir(&contexts, &symbol_table);
                                 println!("Generated LLVM IR for {}.", flow_name);
                                 
                                 // Export the generated module to a .so library artifact
                                 match codegen.emit_library(flow_name) {
                                     Ok(_) => println!("Successfully compiled {}.so library!", flow_name),
                                     Err(e) => eprintln!("Failed to compile library: {}", e),
                                 }
                             },
                             Err(e) => eprintln!("Synthesis error for {}: {:?}", flow_name, e),
                         }
                     }
                 },
                 Err(e) => {
                     eprintln!("Semantic error: {:?}", e);
                 }
             }
             */
        },
        Err(e) => {
             eprintln!("Parse error: {:?}", e);
        }
    }
}
