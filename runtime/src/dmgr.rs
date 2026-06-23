use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use tch::{Device, Kind, Tensor};

pub struct DataManager {
    cache: HashMap<String, Tensor>,
    data_dir: PathBuf,
    pub data_size: (usize, usize),
    pub device: Device,
}

impl DataManager {
    pub fn new(data_dir: PathBuf, device: Option<Device>) -> Self {
        let dev = device.unwrap_or_else(|| {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        });

        let mut dmgr = DataManager {
            cache: HashMap::new(),
            data_dir,
            data_size: (0, 0),
            device: dev,
        };
        let data = dmgr.get_data("close").unwrap();
        dmgr.data_size = (data.size()[0] as usize, data.size()[1] as usize);
        dmgr
    }

    pub fn get_data(&mut self, name: &str) -> Option<Tensor> {
        if let Some(data) = self.cache.get(name) {
            return Some(data.shallow_clone());
        }

        let path = self.data_dir.join(format!("{}.csv.gz", name));
        if !path.exists() {
            panic!("Data file not found: {:?}", path);
        }

        let loaded_data = if let Ok(file) = File::open(&path) {
            let gz = GzDecoder::new(file);
            let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(gz);
            let mut flat_data: Vec<f64> = Vec::new();
            let mut rows = 0;
            let mut cols = 0;

            for result in reader.records() {
                if let Ok(record) = result {
                    if cols == 0 {
                        cols = record.len();
                    } else if record.len() != cols {
                        panic!(
                            "Data columns mismatch at row {}: expected {}, got {}",
                            rows,
                            cols,
                            record.len()
                        );
                    }
                    for field in record.iter() {
                        flat_data.push(field.parse::<f64>().unwrap_or(f64::NAN));
                    }
                    rows += 1;
                }
            }

            if self.data_size.0 != 0 && rows != self.data_size.0 {
                panic!(
                    "Data length mismatch for {}: expected {}, got {}",
                    name, self.data_size.0, rows
                );
            }
            if self.data_size.1 != 0 && cols != self.data_size.1 {
                panic!(
                    "Data columns mismatch for {}: expected {}, got {}",
                    name, self.data_size.1, cols
                );
            }

            if rows == 0 {
                Tensor::zeros(&[0, 0], (Kind::Float, self.device))
            } else {
                println!(
                    "Creating tensor from slice, flat_data.len() = {}, rows = {}, cols = {}",
                    flat_data.len(),
                    rows,
                    cols
                );
                Tensor::from_slice(&flat_data)
                    .view((rows as i64, cols as i64))
                    .to(self.device)
            }
        } else {
            panic!("Failed to load data file: {:?}", path);
        };

        self.cache
            .insert(name.to_string(), loaded_data.shallow_clone());
        Some(loaded_data)
    }
}
