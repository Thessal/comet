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
                     println!("ADTs: {}", analyzer.symbol_table.adts.len());
                     println!("Classes: {}", analyzer.symbol_table.classes.len());
                     println!("Instances: {}", analyzer.symbol_table.instances.len());
                     println!("Functions: {}", analyzer.symbol_table.functions.len());
                     
                     // Synthesis Step
                     let synthesizer = comet::synthesis::Synthesizer::new(&analyzer.symbol_table);
                     // Look for a 'strategy' function or similar entry point.
                     let entry_point = "strategy";
                     if analyzer.symbol_table.functions.contains_key(entry_point) {
                         match synthesizer.synthesize(entry_point) {
                             Ok(contexts) => {
                                 println!("Synthesis successful! Generated {} variants.", contexts.len());
                                 let codegen = comet::codegen::Codegen::new();
                                 let rust_code = codegen.generate_library(&contexts);
                                 println!("--- Generated Rust Library ---");
                                 println!("{}", rust_code);
                                 println!("------------------------------");
                             },
                             Err(e) => eprintln!("Synthesis error: {:?}", e),
                         }
                     } else {
                         println!("No '{}' function found to synthesize.", entry_point);
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
