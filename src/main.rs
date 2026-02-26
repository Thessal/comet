mod comet;

use std::fs;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The input .cm file to compile
    file: String,

    /// The fraction of variants to sample per stage (0.0 to 1.0)
    #[arg(long, default_value_t = 1.0)]
    sample_rate: f64,

    /// Number of exclusive sampling stages to emit as individual .so files
    #[arg(long, default_value_t = 1)]
    exclusive_sample_stages: usize,  //maybe better shortter name could be used
}

fn main() {
    let args = Args::parse();
    let filename = &args.file;
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
                                     
                                     let mut available_variants = real_exprs.clone();
                                     let mut rng = rand::thread_rng();
                                     use rand::seq::SliceRandom;
                                     use rand::distributions::WeightedIndex;
                                     use rand::prelude::Distribution;

                                     for stage in 0..args.exclusive_sample_stages {
                                         let sample_size = std::cmp::min(
                                             (real_exprs.len() as f64 * args.sample_rate).ceil() as usize,
                                             available_variants.len()
                                         );
                                         
                                         if sample_size == 0 {
                                             println!("No more variants to sample. Ending stages early.");
                                             break;
                                         }

                                         let mut weights = Vec::new();
                                         for variant in &available_variants {
                                             weights.push(comet::synthesis::Synthesizer::calculate_forest_weight(variant, &behaviors));
                                         }
                                         
                                         let mut sampled_forests = Vec::new();
                                         if let Ok(dist) = WeightedIndex::new(&weights) {
                                             let mut selected_indices = std::collections::HashSet::new();
                                             while selected_indices.len() < sample_size {
                                                 let idx = dist.sample(&mut rng);
                                                 if selected_indices.insert(idx) {
                                                     sampled_forests.push(available_variants[idx].clone());
                                                 }
                                             }
                                             
                                             let mut remaining = Vec::new();
                                             for (i, v) in available_variants.into_iter().enumerate() {
                                                 if !selected_indices.contains(&i) {
                                                     remaining.push(v);
                                                 }
                                             }
                                             available_variants = remaining;
                                         } else {
                                             available_variants.shuffle(&mut rng);
                                             sampled_forests = available_variants.drain(..sample_size).collect();
                                         }

                                         println!("--- Stage {} ({} samples) ---", stage, sampled_forests.len());
                                         
                                         let graph = comet::ir::ExecutionGraph::from_variants(&sampled_forests);
                                         
                                         let inkwell_ctx = inkwell::context::Context::create();
                                         let module_name = format!("{}_stage_{}", flow.name, stage);
                                         let codegen = comet::codegen::Codegen::new(&inkwell_ctx, &module_name);
                                         
                                         let ir_string = codegen.generate_ir(&graph);
                                         // Print first few lines of IR for brief logging
                                         let ir_preview: String = ir_string.lines().take(5).collect::<Vec<&str>>().join("\n");
                                         println!("Generated LLVM IR for {} (Preview):\n{} ...\n", module_name, ir_preview);
                                         
                                         match codegen.emit_library(&module_name) {
                                              Ok(_) => println!("Successfully compiled {}.so library!", module_name),
                                              Err(e) => eprintln!("Failed to compile library: {}", e),
                                         }
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
