use crate::PartialDeque;
use chrono::NaiveDateTime;
use flate2::read::GzDecoder;
use std::ffi::CStr;
use std::fs::File;
use std::os::raw::c_char;
pub struct DataState {
    pub rows: crate::DequeState,
    pub columns: Vec<String>,
    pub index: Vec<NaiveDateTime>,
    pub current_step: usize,
}

fn resolve_path(id_str: &str, ext: &str) -> String {
    let p1 = format!("./data/{}{}", id_str, ext);
    if std::path::Path::new(&p1).exists() {
        return p1;
    }
    let p2 = format!("comet_examples/data/{}{}", id_str, ext);
    if std::path::Path::new(&p2).exists() {
        return p2;
    }
    p1
}

impl DataState {
    pub fn new(id_str: &str, len: usize) -> Self {
        let file_path = resolve_path(id_str, ".csv.gz");
        let mut row_count = 0;
        if let Ok(file) = File::open(&file_path) {
            let decoder = GzDecoder::new(file);
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(decoder);
            row_count = rdr.records().count();
        }

        let mut rows = crate::DequeState::new(row_count, len);

        if let Ok(file) = File::open(&file_path) {
            let decoder = GzDecoder::new(file);
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(decoder);

            for result in rdr.records() {
                if let Ok(record) = result {
                    let mut row = vec![f64::NAN; len];
                    for (i, field) in record.iter().enumerate().take(len) {
                        if let Ok(val) = field.parse::<f64>() {
                            row[i] = val;
                        }
                    }
                    rows.push(&row);
                }
            }
        }

        let mut columns = Vec::new();
        let cols_path = format!("./data/{}.columns.csv.gz", id_str);
        if let Ok(file) = File::open(&cols_path) {
            let decoder = GzDecoder::new(file);
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(decoder);
            if let Some(result) = rdr.records().next() {
                if let Ok(record) = result {
                    for field in record.iter() {
                        columns.push(field.to_string());
                    }
                }
            }
        }

        let mut index = Vec::new();
        let idx_path = format!("./data/{}.index.csv.gz", id_str);
        if let Ok(file) = File::open(&idx_path) {
            let decoder = GzDecoder::new(file);
            let mut rdr = csv::ReaderBuilder::new()
                .has_headers(false)
                .from_reader(decoder);
            for result in rdr.records() {
                if let Ok(record) = result {
                    if let Some(field) = record.get(0) {
                        if let Ok(dt) = NaiveDateTime::parse_from_str(field, "%Y-%m-%d %H:%M:%S") {
                            index.push(dt);
                        } else {
                            index.push(NaiveDateTime::default());
                        }
                    }
                }
            }
        }

        DataState {
            rows,
            columns,
            index,
            current_step: 0,
        }
    }

    pub fn step(&mut self, out_ptr: *mut f64, len: usize) {
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, len) };
        if self.current_step < self.rows.count {
            let row_slice = &self.rows.history[self.current_step];

            for i in 0..len {
                if i < row_slice.len() {
                    out_slice[i] = row_slice[i];
                } else {
                    out_slice[i] = f64::NAN;
                }
            }
            self.current_step += 1;
        } else {
            for i in 0..len {
                out_slice[i] = f64::NAN;
            }
        }
    }

    pub fn drop_buffers(&mut self) {
        self.columns.clear();
        self.index.clear();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_data_init(
    id_ptr: *const c_char,
    id_len: usize,
    len: usize,
) -> *mut DataState {
    let id_str = if id_ptr.is_null() {
        String::new()
    } else {
        unsafe {
            if id_len > 0 {
                let slice = std::slice::from_raw_parts(id_ptr as *const u8, id_len);
                String::from_utf8_lossy(slice).into_owned()
            } else {
                CStr::from_ptr(id_ptr).to_string_lossy().into_owned()
            }
        }
    };

    let state = Box::new(DataState::new(&id_str, len));
    Box::into_raw(state)
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_data_free(state: *mut DataState) {
    if !state.is_null() {
        unsafe {
            let mut s = Box::from_raw(state);
            s.drop_buffers();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn comet_data_step(state: *mut DataState, out_ptr: *mut f64, len: usize) {
    let s = unsafe { &mut *state };
    s.step(out_ptr, len)
}
