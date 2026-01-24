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
             
             let mut analyzer = comet::semantics::SemanticAnalyzer::new();
             match analyzer.analyze(&program, filename) {
                 Ok(_) => {
                     println!("Semantic analysis passed!");
                     println!("Symbol Table Stats:");
                     println!("Types: {}", analyzer.symbol_table.types.len());
                     println!("Behaviors: {}", analyzer.symbol_table.behaviors.len());
                     println!("Impls: {}", analyzer.symbol_table.implementations.len());
                     println!("Functions: {}", analyzer.symbol_table.functions.len());
                     println!("Flows: {}", analyzer.symbol_table.flows.len());
                     
                     // Synthesis Step
                     let synthesizer = comet::synthesis::Synthesizer::new(&analyzer.symbol_table);
                     // Assuming 'Strategy' flow exists for now, or we can iterate
                     if let Some(_) = analyzer.symbol_table.flows.get("Strategy") {
                         match synthesizer.synthesize("Strategy") {
                             Ok(contexts) => {
                                 println!("Synthesis successful!");
                                 for (i, ctx) in contexts.iter().enumerate() {
                                     println!("Context {}:", i);
                                     println!("{:#?}", ctx);
                                 }
                             },
                             Err(e) => eprintln!("Synthesis error: {:?}", e),
                         }
                     } else {
                         println!("No 'Strategy' flow found to synthesize.");
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
