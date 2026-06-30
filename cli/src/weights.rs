use tch::nn::VarStore;

pub fn save(vs: &VarStore, path: &Option<String>) {
    if let Some(p) = path {
        if let Some(parent) = std::path::Path::new(p).parent() {
            if !parent.exists() && parent != std::path::Path::new("") {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    eprintln!("Failed to create directory {:?}: {:?}", parent, e);
                }
            }
        }
        if let Err(e) = vs.save(p) {
            eprintln!("Failed to save weights to {}: {:?}", p, e);
        } else {
            println!("Weights saved to {}", p);
        }
    } else {
        println!("No weights path specified. Skipping save.");
    }
}

pub fn load(vs: &mut VarStore, path: &Option<String>) {
    if let Some(p) = path {
        if std::path::Path::new(p).exists() {
            if let Err(e) = vs.load(p) {
                eprintln!("Failed to load weights from {}: {:?}", p, e);
            } else {
                println!("Weights loaded from {}", p);
            }
        } else {
            println!("Weight file {} does not exist. Skipping load.", p);
        }
    }
}