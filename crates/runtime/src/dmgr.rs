use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use stdlib::types::get_size;
use tch::{Device, Kind, Tensor};

pub const CMT_MAGIC: &[u8; 4] = b"CMT1";
pub const CMT_VERSION: u32 = 1;
pub const CMT_DTYPE_F32: u32 = 1;
pub const CMT_DTYPE_F64: u32 = 2;
pub const CMT_HEADER_SIZE: usize = 64;

/// Controls whether `.bin` binary files are loaded via POSIX `mmap` or read into heap memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// Memory-map `.bin` files directly from the OS page cache (`mmap`).
    Mmap,
    /// Read the `.bin` file bytes into a heap buffer (`std::fs::read`).
    InMemory,
}

/// Deterministic 64-bit FNV-1a hash of a list of coordinate strings (dates or instruments).
pub fn hash_coords(items: &[String]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for item in items {
        for &b in item.as_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Represents the target backtest universe `(dates, instruments)` and fast lookup maps.
#[derive(Debug, Clone)]
pub struct Universe {
    pub dates: Vec<String>,
    pub instruments: Vec<String>,
    pub date_to_idx: HashMap<String, usize>,
    pub inst_to_idx: HashMap<String, usize>,
    pub dates_hash: u64,
    pub insts_hash: u64,
}

impl Universe {
    pub fn new(dates: Vec<String>, instruments: Vec<String>) -> Self {
        let date_to_idx = dates
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let inst_to_idx = instruments
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let dates_hash = hash_coords(&dates);
        let insts_hash = hash_coords(&instruments);
        Self {
            dates,
            instruments,
            date_to_idx,
            inst_to_idx,
            dates_hash,
            insts_hash,
        }
    }

    /// Discovers universe dates and instruments from `data_dir` (or fallback `wrds/`).
    pub fn discover(data_dir: &Path) -> Option<Self> {
        let mut dates: Option<Vec<String>> = None;
        let mut instruments: Option<Vec<String>> = None;

        // 1. Check info.json or universe.json in data_dir
        for json_name in ["universe.json", "info.json"] {
            let p = data_dir.join(json_name);
            if let Ok(content) = fs::read_to_string(&p) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if dates.is_none() {
                        if let Some(arr) = val.get("dates").and_then(|v| v.as_array()) {
                            dates = Some(
                                arr.iter()
                                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                                    .collect(),
                            );
                        } else if let Some(df_path) = val.get("dates_file").and_then(|v| v.as_str())
                        {
                            dates = Self::read_lines(&data_dir.join(df_path))
                                .or_else(|| Self::read_lines(Path::new(df_path)));
                        }
                    }
                    if instruments.is_none() {
                        if let Some(arr) = val.get("instruments").and_then(|v| v.as_array()) {
                            instruments = Some(
                                arr.iter()
                                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                                    .collect(),
                            );
                        } else if let Some(if_path) =
                            val.get("instruments_file").and_then(|v| v.as_str())
                        {
                            instruments = Self::read_lines(&data_dir.join(if_path))
                                .or_else(|| Self::read_lines(Path::new(if_path)));
                        }
                    }
                }
            }
        }

        // 2. Check universe_dates.csv / universe_instruments.csv in data_dir or wrds/
        let candidate_dirs = [
            data_dir.to_path_buf(),
            data_dir.join("../wrds"),
            PathBuf::from("wrds"),
        ];
        if dates.is_none() {
            for d in &candidate_dirs {
                if let Some(lines) = Self::read_lines(&d.join("universe_dates.csv")) {
                    dates = Some(lines);
                    break;
                }
            }
        }
        if instruments.is_none() {
            for d in &candidate_dirs {
                if let Some(lines) = Self::read_lines(&d.join("universe_instruments.csv")) {
                    instruments = Some(lines);
                    break;
                }
            }
        }

        match (dates, instruments) {
            (Some(d), Some(i)) if !d.is_empty() && !i.is_empty() => Some(Self::new(d, i)),
            _ => None,
        }
    }

    fn read_lines(path: &Path) -> Option<Vec<String>> {
        let content = fs::read_to_string(path).ok()?;
        let lines: Vec<String> = content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if lines.is_empty() {
            None
        } else {
            Some(lines)
        }
    }
}

/// Safe RAII wrapper around POSIX `mmap` for read-only `.cmt` files.
struct MmapBuffer {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for MmapBuffer {}
unsafe impl Sync for MmapBuffer {}

impl MmapBuffer {
    fn map_file(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {:?}: {}", path, e))?;
        let len = file
            .metadata()
            .map_err(|e| format!("Failed to stat {:?}: {}", path, e))?
            .len() as usize;
        if len < CMT_HEADER_SIZE {
            return Err(format!("File {:?} is smaller than 64-byte CMT header", path));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap failed on {:?}", path));
        }
        Ok(Self {
            ptr: ptr as *const u8,
            len,
        })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for MmapBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct CmtHeader {
    pub version: u32,
    pub dtype: u32,
    pub flags: u32,
    pub rows: usize,
    pub cols: usize,
    pub data_offset: usize,
    pub meta_len: usize,
    pub dates_hash: u64,
    pub insts_hash: u64,
}

impl CmtHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < CMT_HEADER_SIZE {
            return Err("Buffer too short for CMT1 header".to_string());
        }
        if &bytes[0..4] != CMT_MAGIC {
            return Err(format!("Invalid CMT magic bytes: {:?}", &bytes[0..4]));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let dtype = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        let rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        let cols = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
        let data_offset = u64::from_le_bytes(bytes[32..40].try_into().unwrap()) as usize;
        let meta_len = u64::from_le_bytes(bytes[40..48].try_into().unwrap()) as usize;
        let dates_hash = u64::from_le_bytes(bytes[48..56].try_into().unwrap());
        let insts_hash = u64::from_le_bytes(bytes[56..64].try_into().unwrap());

        if version != CMT_VERSION {
            return Err(format!("Unsupported CMT version: {}", version));
        }
        if data_offset < CMT_HEADER_SIZE + meta_len || data_offset > bytes.len() {
            return Err(format!(
                "Invalid CMT data_offset={} (meta_len={}, total_len={})",
                data_offset,
                meta_len,
                bytes.len()
            ));
        }
        let elem_size = match dtype {
            CMT_DTYPE_F32 => 4usize,
            CMT_DTYPE_F64 => 8usize,
            _ => return Err(format!("Unsupported CMT dtype code: {}", dtype)),
        };
        let expected_payload = rows.checked_mul(cols).and_then(|n| n.checked_mul(elem_size)).unwrap_or(usize::MAX);
        if bytes.len() < data_offset + expected_payload {
            return Err(format!(
                "Truncated CMT payload: expected {} bytes at offset {}, file has {}",
                expected_payload,
                data_offset,
                bytes.len()
            ));
        }
        Ok(Self {
            version,
            dtype,
            flags,
            rows,
            cols,
            data_offset,
            meta_len,
            dates_hash,
            insts_hash,
        })
    }
}

pub struct DataManager {
    cache: HashMap<String, Tensor>,
    data_dir: PathBuf,
    pub data_size: (usize, usize),
    pub device: Device,
    pub load_mode: LoadMode,
    pub universe: Option<Universe>,
}

impl DataManager {
    pub fn new(data_dir: PathBuf, device: Option<Device>) -> Self {
        Self::with_mode(data_dir, device, LoadMode::Mmap)
    }

    pub fn with_mode(data_dir: PathBuf, device: Option<Device>, load_mode: LoadMode) -> Self {
        let dev = device.unwrap_or_else(|| {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        });

        let universe = Universe::discover(&data_dir);
        let sz = get_size();
        let data_size = if sz[0] > 0 && sz[1] > 0 {
            (sz[0] as usize, sz[1] as usize)
        } else if let Some(ref u) = universe {
            (u.dates.len(), u.instruments.len())
        } else {
            (0, 0)
        };

        DataManager {
            cache: HashMap::new(),
            data_dir,
            data_size,
            device: dev,
            load_mode,
            universe,
        }
    }

    /// Sets an explicit target universe `(dates, instruments)` for on-the-fly alignment.
    pub fn set_universe(&mut self, dates: Vec<String>, instruments: Vec<String>) {
        self.data_size = (dates.len(), instruments.len());
        stdlib::types::set_size(dates.len() as i64, instruments.len() as i64);
        self.universe = Some(Universe::new(dates, instruments));
        self.cache.clear();
    }

    pub fn get_data(&mut self, name: &str) -> Option<Tensor> {
        if let Some(data) = self.cache.get(name) {
            return Some(data.shallow_clone());
        }

        // 1. Prefer binary `.bin` format if available
        let bin_path = self.data_dir.join(format!("{}.bin", name));
        let loaded_data = if bin_path.exists() {
            self.load_bin(&bin_path, name)
        } else {
            let csv_gz_path = self.data_dir.join(format!("{}.csv.gz", name));
            if !csv_gz_path.exists() {
                panic!(
                    "Data file not found: neither {:?} nor {:?} exists",
                    bin_path, csv_gz_path
                );
            }
            self.load_csv_gz(&csv_gz_path, name)
        };

        self.cache
            .insert(name.to_string(), loaded_data.shallow_clone());
        Some(loaded_data)
    }

    fn load_bin(&self, path: &Path, name: &str) -> Tensor {
        match self.load_mode {
            LoadMode::Mmap => {
                let mmap = MmapBuffer::map_file(path)
                    .unwrap_or_else(|e| panic!("Failed to mmap {:?}: {}", path, e));
                self.parse_and_align_bin(mmap.as_slice(), name)
            }
            LoadMode::InMemory => {
                let mut file = File::open(path)
                    .unwrap_or_else(|e| panic!("Failed to open {:?}: {}", path, e));
                let mut buf = Vec::new();
                file.read_to_end(&mut buf)
                    .unwrap_or_else(|e| panic!("Failed to read {:?}: {}", path, e));
                self.parse_and_align_bin(&buf, name)
            }
        }
    }

    fn parse_and_align_bin(&self, bytes: &[u8], name: &str) -> Tensor {
        let hdr = CmtHeader::parse(bytes)
            .unwrap_or_else(|e| panic!("Corrupt CMT header in {}: {}", name, e));
        let payload = &bytes[hdr.data_offset..];

        // Check whether the CMT grid already matches our target universe
        let is_exact_match = if let Some(ref u) = self.universe {
            hdr.rows == u.dates.len()
                && hdr.cols == u.instruments.len()
                && (hdr.dates_hash == 0 || hdr.dates_hash == u.dates_hash)
                && (hdr.insts_hash == 0 || hdr.insts_hash == u.insts_hash)
        } else {
            (self.data_size.0 == 0 || hdr.rows == self.data_size.0)
                && (self.data_size.1 == 0 || hdr.cols == self.data_size.1)
        };

        if is_exact_match {
            let total = hdr.rows * hdr.cols;
            if total == 0 {
                return Tensor::zeros(&[0, 0], (Kind::Float, self.device));
            }
            // Fast path: if 8-byte / 4-byte aligned, slice directly from mmap/buffer without extra copy
            return match hdr.dtype {
                CMT_DTYPE_F64 => {
                    if (payload.as_ptr() as usize) % std::mem::align_of::<f64>() == 0 {
                        let slice = unsafe {
                            std::slice::from_raw_parts(payload.as_ptr() as *const f64, total)
                        };
                        Tensor::from_slice(slice)
                            .view((hdr.rows as i64, hdr.cols as i64))
                            .to(self.device)
                    } else {
                        let vec: Vec<f64> = payload[..total * 8]
                            .chunks_exact(8)
                            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        Tensor::from_slice(&vec)
                            .view((hdr.rows as i64, hdr.cols as i64))
                            .to(self.device)
                    }
                }
                CMT_DTYPE_F32 => {
                    let vec: Vec<f64> = payload[..total * 4]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                        .collect();
                    Tensor::from_slice(&vec)
                        .view((hdr.rows as i64, hdr.cols as i64))
                        .to(self.device)
                }
                _ => unreachable!(),
            };
        }

        // Dynamic Alignment Path: parse embedded JSON coordinates and reindex onto `self.universe`
        let target_univ = self.universe.as_ref().unwrap_or_else(|| {
            panic!(
                "CMT file '{}' has shape ({}, {}) which differs from expected {:?}, and no target Universe is loaded for alignment",
                name, hdr.rows, hdr.cols, self.data_size
            )
        });

        let meta_bytes = &bytes[CMT_HEADER_SIZE..CMT_HEADER_SIZE + hdr.meta_len];
        let meta: serde_json::Value = serde_json::from_slice(meta_bytes).unwrap_or_else(|e| {
            panic!(
                "Failed to parse embedded JSON metadata in CMT '{}': {}",
                name, e
            )
        });

        let src_dates: Vec<String> = meta["dates"]
            .as_array()
            .unwrap_or_else(|| panic!("CMT '{}' metadata missing 'dates' array", name))
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        let src_insts: Vec<String> = meta["instruments"]
            .as_array()
            .unwrap_or_else(|| panic!("CMT '{}' metadata missing 'instruments' array", name))
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        assert_eq!(
            src_dates.len(),
            hdr.rows,
            "CMT '{}' metadata dates len != header rows",
            name
        );
        assert_eq!(
            src_insts.len(),
            hdr.cols,
            "CMT '{}' metadata instruments len != header cols",
            name
        );

        let dst_rows = target_univ.dates.len();
        let dst_cols = target_univ.instruments.len();
        let mut aligned = vec![f64::NAN; dst_rows * dst_cols];

        // Precompute column mapping: src_col -> Option<dst_col>
        let col_map: Vec<Option<usize>> = src_insts
            .iter()
            .map(|inst| target_univ.inst_to_idx.get(inst).copied())
            .collect();

        for (src_r, date_str) in src_dates.iter().enumerate() {
            if let Some(&dst_r) = target_univ.date_to_idx.get(date_str) {
                let dst_row_offset = dst_r * dst_cols;
                let src_row_offset = src_r * hdr.cols;
                for (src_c, maybe_dst_c) in col_map.iter().enumerate() {
                    if let Some(dst_c) = maybe_dst_c {
                        let val = match hdr.dtype {
                            CMT_DTYPE_F64 => {
                                let off = (src_row_offset + src_c) * 8;
                                f64::from_le_bytes(payload[off..off + 8].try_into().unwrap())
                            }
                            CMT_DTYPE_F32 => {
                                let off = (src_row_offset + src_c) * 4;
                                f32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as f64
                            }
                            _ => unreachable!(),
                        };
                        aligned[dst_row_offset + dst_c] = val;
                    }
                }
            }
        }

        Tensor::from_slice(&aligned)
            .view((dst_rows as i64, dst_cols as i64))
            .to(self.device)
    }

    fn load_csv_gz(&self, path: &Path, name: &str) -> Tensor {
        if let Ok(file) = File::open(path) {
            let gz = GzDecoder::new(file);
            let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(gz);
            let mut records: Vec<csv::StringRecord> = Vec::new();
            for res in reader.records() {
                if let Ok(rec) = res {
                    records.push(rec);
                }
            }

            if records.is_empty() {
                return Tensor::zeros(&[0, 0], (Kind::Float, self.device));
            }

            // Detect whether CSV has labeled headers (row 0 = instrument IDs, col 0 = dates)
            let first_cell = records[0].get(0).unwrap_or("").trim();
            let has_header_and_index = first_cell.parse::<f64>().is_err() && !first_cell.is_empty()
                || (records[0].len() > 1
                    && records[0].get(1).unwrap_or("").parse::<f64>().is_err()
                    && !records[0].get(1).unwrap_or("").is_empty());

            if has_header_and_index {
                let target_univ = self.universe.as_ref().unwrap_or_else(|| {
                    panic!(
                        "Labeled CSV {:?} requires a target Universe to align dates and instruments",
                        path
                    )
                });
                let dst_rows = target_univ.dates.len();
                let dst_cols = target_univ.instruments.len();
                let mut aligned = vec![f64::NAN; dst_rows * dst_cols];

                let header = &records[0];
                let col_map: Vec<Option<usize>> = (1..header.len())
                    .map(|c| {
                        let inst = header.get(c).unwrap_or("").trim();
                        target_univ.inst_to_idx.get(inst).copied()
                    })
                    .collect();

                for rec in records.iter().skip(1) {
                    let date_str = rec.get(0).unwrap_or("").trim();
                    if let Some(&dst_r) = target_univ.date_to_idx.get(date_str) {
                        let dst_row_offset = dst_r * dst_cols;
                        for (idx, maybe_dst_c) in col_map.iter().enumerate() {
                            if let Some(dst_c) = maybe_dst_c {
                                let val_str = rec.get(idx + 1).unwrap_or("").trim();
                                let val = if val_str.is_empty() {
                                    f64::NAN
                                } else {
                                    val_str.parse::<f64>().unwrap_or(f64::NAN)
                                };
                                aligned[dst_row_offset + dst_c] = val;
                            }
                        }
                    }
                }
                return Tensor::from_slice(&aligned)
                    .view((dst_rows as i64, dst_cols as i64))
                    .to(self.device);
            }

            // Standard headerless matrix CSV
            let rows = records.len();
            let cols = records[0].len();
            let mut flat_data: Vec<f64> = Vec::with_capacity(rows * cols);

            for (r_idx, record) in records.iter().enumerate() {
                if record.len() != cols {
                    panic!(
                        "Data columns mismatch at row {}: expected {}, got {}",
                        r_idx,
                        cols,
                        record.len()
                    );
                }
                for field in record.iter() {
                    flat_data.push(field.parse::<f64>().unwrap_or(f64::NAN));
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

            println!(
                "Creating tensor from slice, flat_data.len() = {}, rows = {}, cols = {}",
                flat_data.len(),
                rows,
                cols
            );
            Tensor::from_slice(&flat_data)
                .view((rows as i64, cols as i64))
                .to(self.device)
        } else {
            panic!("Failed to load data file: {:?}", path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_cmt(
        path: &Path,
        dates: &[&str],
        insts: &[&str],
        values: &[f64],
    ) {
        let rows = dates.len();
        let cols = insts.len();
        assert_eq!(values.len(), rows * cols);

        let d_vec: Vec<String> = dates.iter().map(|s| s.to_string()).collect();
        let i_vec: Vec<String> = insts.iter().map(|s| s.to_string()).collect();
        let d_hash = hash_coords(&d_vec);
        let i_hash = hash_coords(&i_vec);

        let meta_json = serde_json::json!({
            "dates": dates,
            "instruments": insts,
        })
        .to_string();
        let meta_bytes = meta_json.as_bytes();
        let meta_len = meta_bytes.len();
        let raw_end = CMT_HEADER_SIZE + meta_len;
        let data_offset = (raw_end + 63) & !63;

        let mut buf = vec![0u8; data_offset + values.len() * 8];
        buf[0..4].copy_from_slice(CMT_MAGIC);
        buf[4..8].copy_from_slice(&CMT_VERSION.to_le_bytes());
        buf[8..12].copy_from_slice(&CMT_DTYPE_F64.to_le_bytes());
        buf[12..16].copy_from_slice(&1u32.to_le_bytes());
        buf[16..24].copy_from_slice(&(rows as u64).to_le_bytes());
        buf[24..32].copy_from_slice(&(cols as u64).to_le_bytes());
        buf[32..40].copy_from_slice(&(data_offset as u64).to_le_bytes());
        buf[40..48].copy_from_slice(&(meta_len as u64).to_le_bytes());
        buf[48..56].copy_from_slice(&d_hash.to_le_bytes());
        buf[56..64].copy_from_slice(&i_hash.to_le_bytes());
        buf[CMT_HEADER_SIZE..CMT_HEADER_SIZE + meta_len].copy_from_slice(meta_bytes);

        for (idx, &v) in values.iter().enumerate() {
            let off = data_offset + idx * 8;
            buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }

        let mut f = File::create(path).unwrap();
        f.write_all(&buf).unwrap();
    }

    #[test]
    fn test_cmt_mmap_and_universe_alignment() {
        let tmp_dir = std::env::temp_dir().join("comet_dmgr_test");
        let _ = fs::remove_dir_all(&tmp_dir);
        fs::create_dir_all(&tmp_dir).unwrap();

        // Source CMT has 3 dates and 3 instruments in scrambled order
        let src_dates = ["2020-01-01", "2020-01-02", "2020-01-03"];
        let src_insts = ["INST_B", "INST_A", "INST_C"];
        let src_vals = [
            10.0, 20.0, 30.0, // 2020-01-01
            11.0, 21.0, 31.0, // 2020-01-02
            12.0, 22.0, 32.0, // 2020-01-03
        ];
        write_test_cmt(&tmp_dir.join("test_field.bin"), &src_dates, &src_insts, &src_vals);

        // Target universe has 2 dates (including one missing in source) and 2 instruments [INST_A, INST_D (missing)]
        let mut dmgr = DataManager::with_mode(tmp_dir.clone(), Some(Device::Cpu), LoadMode::Mmap);
        dmgr.set_universe(
            vec!["2020-01-02".into(), "2020-01-04".into()],
            vec!["INST_A".into(), "INST_B".into(), "INST_D".into()],
        );

        let tensor = dmgr.get_data("test_field").unwrap();
        assert_eq!(tensor.size(), vec![2, 3]);
        let flat: Vec<f64> = Vec::<f64>::try_from(tensor.flatten(0, -1)).unwrap();
        // Row 0 ("2020-01-02"): INST_A = 21.0, INST_B = 11.0, INST_D = NaN
        assert_eq!(flat[0], 21.0);
        assert_eq!(flat[1], 11.0);
        assert!(flat[2].is_nan());
        // Row 1 ("2020-01-04"): all NaN
        assert!(flat[3].is_nan());
        assert!(flat[4].is_nan());
        assert!(flat[5].is_nan());

        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
