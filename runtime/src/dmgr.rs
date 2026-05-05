use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

pub struct DataManager {
    cache: HashMap<String, Vec<Vec<f64>>>,
    data_dir: PathBuf,
}

impl DataManager {
    pub fn new(data_dir: PathBuf) -> Self {
        DataManager {
            cache: HashMap::new(),
            data_dir,
        }
    }

    pub fn get_data(&mut self, name: &str) -> Option<Vec<Vec<f64>>> {
        if let Some(data) = self.cache.get(name) {
            return Some(data.clone());
        }

        let filename1 = self.data_dir.join(format!("{}.csv.gz", name));

        let path = if filename1.exists() {
            filename1
        } else {
            // fallback for tests
            PathBuf::from(format!("../data/{}.csv.gz", name))
        };

        let loaded_data = if let Ok(file) = File::open(&path) {
            let gz = GzDecoder::new(file);
            let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(gz);
            let mut rows: Vec<Vec<f64>> = Vec::new();

            for result in reader.records() {
                if let Ok(record) = result {
                    rows.push(
                        record
                            .iter()
                            .map(|field| field.parse::<f64>().unwrap_or(f64::NAN))
                            .collect(),
                    );
                }
            }
            if rows.is_empty() { vec![vec![]] } else { rows }
        } else {
            // panic!("Failed to load data file: {:?}", path);
            return None;
        };

        self.cache.insert(name.to_string(), loaded_data.clone());
        Some(loaded_data)
    }
}
