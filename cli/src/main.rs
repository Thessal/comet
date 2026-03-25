
use clap::Parser;
use std::fs;

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
    sample_stages: usize,

    /// Optional path to the stdlib source directory or .so file
    #[arg(long)]
    stdlib: Option<String>,
}

fn main() {
    let args = Args::parse();
    let filename = &args.file;
    let content = fs::read_to_string(filename).expect("Failed to read file");

    match parser::parser::parse(&content) {
        Ok(mut program) => {
            println!("Parsed successfully!");
            println!("--- AST Representation ---");
            println!("{}", program);
            println!("--------------------------");
            println!("Total declarations: {}", program.declarations.len());
            for d in &program.declarations {
                let t = match d {
                    parser::ast::Declaration::Import(_) => "Import",
                    parser::ast::Declaration::Behavior(_) => "Behavior",

                    parser::ast::Declaration::Flow(_) => "Flow",
                };
                println!("Decl: {}", t);
            }
            // Handle simple imports
            let mut all_decls = Vec::new();
            for decl in &program.declarations {
                if let parser::ast::Declaration::Import(imp) = decl {
                    let mut import_path = std::path::PathBuf::from(filename);
                    import_path.pop();
                    let import_str = imp.path.replace("\"", "").replace(" ", "");
                    import_path.push(import_str);

                    if let Ok(imp_content) = fs::read_to_string(&import_path) {
                        match parser::parser::parse(&imp_content) {
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
                    parser::ast::Declaration::Behavior(b) => {
                        behaviors.insert(b.name.clone(), b.clone());
                    }
                    _ => {}
                }
            }

            // Build the standard function library
            for (name, param_types, ret_type) in rl::env::get_available_funcs() {
                let mut args = Vec::new();
                for (i, p_type) in param_types.into_iter().enumerate() {
                    args.push((format!("arg{}", i), p_type));
                }
                library.push(rl::synthesis::FnSignature {
                    name,
                    args,
                    return_type: ret_type,
                });
            }

            println!("Evaluating flows directly since semantics module is removed for redesign:");
            for decl in &program.declarations {
                if let parser::ast::Declaration::Flow(flow) = decl {
                    println!("Synthesizing flow: {}", flow.name);
                    println!("--- AST Flow Body ---");
                    for stmt in &flow.body {
                        println!("{:#?}", stmt);
                    }
                    println!("---------------------");

                    let mut env_map = std::collections::HashMap::new();
                    for stmt in &flow.body {
                        if let parser::ast::FlowStmt::Assignment { target, expr } = stmt {
                            env_map.insert(target.clone(), expr.clone());
                        }
                    }

                    for stmt in &flow.body {
                        if let parser::ast::FlowStmt::Expr(expr) = stmt {
                            let substituted_expr =
                                rl::synthesis::substitute_expr(expr, &env_map);
                            println!(
                                "Synthesizing fully substituted target expression: {:#?}",
                                substituted_expr
                            );
                            match rl::synthesis::Synthesizer::synthesize_expr(
                                &substituted_expr,
                                &behaviors,
                                &library,
                            ) {
                                Ok(real_exprs) => {
                                    let mut available_variants = real_exprs.clone();
                                    let mut rng = rand::thread_rng();
                                    use rand::distributions::WeightedIndex;
                                    use rand::prelude::Distribution;
                                    use rand::seq::SliceRandom;

                                    for stage in 0..args.sample_stages {
                                        let sample_size = std::cmp::min(
                                            (real_exprs.len() as f64 * args.sample_rate).ceil()
                                                as usize,
                                            available_variants.len(),
                                        );

                                        if sample_size == 0 {
                                            println!(
                                                "No more variants to sample. Ending stages early."
                                            );
                                            break;
                                        }

                                        available_variants.shuffle(&mut rng);
                                        let sampled_forests: Vec<_> =
                                            available_variants.drain(..sample_size).collect();
                                        println!(
                                            "--- Stage {} ({} samples) ---",
                                            stage,
                                            sampled_forests.len()
                                        );

                                        let graph = codegen::ir::ExecutionGraph::from_variants(
                                            &sampled_forests,
                                        );

                                        let module_name = format!("{}_stage_{}", flow.name, stage);
                                        let codegen = codegen::codegen::Codegen::new(
                                            &module_name,
                                            args.stdlib.clone(),
                                        );

                                        let ir_string = codegen.generate_ir(&graph);
                                        // Print first few lines of IR for brief logging
                                        let ir_preview: String = ir_string
                                            .lines()
                                            .take(5)
                                            .collect::<Vec<&str>>()
                                            .join("\n");
                                        println!(
                                            "Generated Rust source for {} (Preview):\n{} ...\n",
                                            module_name, ir_preview
                                        );

                                        match codegen.emit_library(&module_name, &ir_string) {
                                            Ok(_) => println!(
                                                "Successfully compiled {}.so library!",
                                                module_name
                                            ),
                                            Err(e) => eprintln!("Failed to compile library: {}", e),
                                        }
                                    }
                                }
                                Err(e) => eprintln!("Synthesis error: {}", e),
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Parse error: {:?}", e);
        }
    }
}
