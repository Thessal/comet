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
        Ok(program) => {
             println!("Parsed successfully!");
             // println!("{:#?}", program);
             
             // let mut analyzer = comet::semantics::SemanticAnalyzer::new();
             // match analyzer.analyze(&program, filename) {
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
                                 // let codegen = comet::codegen::Codegen::new();
                                 // let rust_code = codegen.generate_library(&contexts);
                                 // println!("--- Generated Rust Library ---");
                                 // println!("{}", rust_code);
                                 // println!("------------------------------");
                             },
                             Err(e) => eprintln!("Synthesis error for {}: {:?}", flow_name, e),
                         }
                     }
                 },
                 Err(e) => {
                     eprintln!("Semantic error: {:?}", e);
                 }
             }
        },
        Err(e) => {
             eprintln!("Parse error: {:?}", e);
        }
    }
}
