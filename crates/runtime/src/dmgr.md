# Comet Memory-Mapped Tensor Format (`.bin`) Specification

## 1. Motivation & Design Goals

In multi-source quantitative research (combining daily OHLCV, quarterly/annual fundamentals, alternative data, and global universes with differing exchange calendars), `.csv.gz` matrices suffer from four fundamental limitations:

1. **No Embedded Coordinate System**: A headerless `(T, N)` CSV relies on external files (`universe_dates.csv`, `universe_instruments.csv`) staying strictly synchronized in row/column order. If a vendor dataset has a different date coverage or instrument subset, loading it without explicit coordinate alignment silently corrupts cross-sectional and time-series alignment.
2. **CPU Decompression & Parsing Bottleneck**: Decompressing `gzip` streams and running `str::parse::<f64>()` millions of times per backtest startup dominates initialization latency and prevents multiple worker processes from sharing memory.
3. **No Zero-Copy `mmap` Support**: Compressed text cannot be mapped directly into virtual memory (`mmap`) or passed directly to LibTorch (`tch::Tensor`) / NumPy / Pandas without full heap allocation.
4. **Inability to Mix Dense Grids & Sparse / Partial-Coverage Sources**: Different data vendors cover different date ranges `[t_start, t_end]` and different instrument subsets `{inst_1, inst_2, ...}`.

The **Comet Memory-Mapped Binary Tensor (`.bin`)** format solves all four problems in a single self-contained, 64-byte aligned binary file.

---

## 2. Binary File Layout (`CMT1`)

Every `.bin` file consists of three regions:
1. **Fixed 64-byte Header** (`offset 0..64`)
2. **JSON Coordinate Metadata Header** (`offset 64 .. data_offset`), padded with trailing spaces/`0x00` so `data_offset` is a multiple of **64 bytes** (cache-line and AVX-512 aligned).
3. **Contiguous Row-Major Binary Tensor Payload** (`offset data_offset .. EOF`), laid out as `[rows, cols]` in little-endian IEEE-754 floating point (`f64` or `f32`).

```text
+-----------------------------------------------------------------------+
| Offset 0..64: Fixed 64-Byte Binary Header (Little-Endian)             |
|   0..4   : Magic bytes b"CMT1"                                        |
|   4..8   : version (u32 = 1)                                          |
|   8..12  : dtype   (u32: 1 = f32, 2 = f64)                            |
|  12..16  : flags   (u32: bit 0 = has_embedded_coords, etc.)           |
|  16..24  : rows    (u64: T, number of dates/timestamps)               |
|  24..32  : cols    (u64: N, number of instruments)                    |
|  32..40  : data_offset (u64: byte offset to payload, 64-byte aligned) |
|  40..48  : meta_len    (u64: exact byte length of JSON metadata)      |
|  48..56  : dates_hash  (u64: FNV-1a 64-bit hash of dates list)        |
|  56..64  : insts_hash  (u64: FNV-1a 64-bit hash of instruments list)  |
+-----------------------------------------------------------------------+
| Offset 64..64+meta_len: UTF-8 JSON Coordinate Metadata                |
|   {                                                                   |
|     "dates": ["2015-02-02", "2015-02-03", ...],                       |
|     "instruments": ["AGP8696W1045", "BMG0585R1060", ...],             |
|     "source": "wrds_basedata",                                        |
|     "frequency": "1d"                                                 |
|   }                                                                   |
+-----------------------------------------------------------------------+
| Offset 64+meta_len..data_offset: Zero padding (0..63 bytes)           |
+-----------------------------------------------------------------------+
| Offset data_offset..EOF: Contiguous Little-Endian Tensor Payload      |
|   f64 (dtype=2): rows * cols * 8 bytes                                |
|   f32 (dtype=1): rows * cols * 4 bytes                                |
+-----------------------------------------------------------------------+
```

### Fast-Path Hash Check (`dates_hash` & `insts_hash`)
Both the Rust `DataManager` and Python `DataManager` compute a deterministic 64-bit FNV-1a hash of the target universe's `dates` list and `instruments` list:
- **Fast Path ($O(1)$ Zero-Copy)**: When `header.dates_hash == target_universe.dates_hash` and `header.insts_hash == target_universe.insts_hash`, the file's `(rows, cols)` grid is **guaranteed** to be 100% aligned with the backtest universe.
  - In `mmap` mode (CPU), the payload at `&mmap[data_offset..]` can be wrapped or sliced directly without parsing the JSON coordinate arrays or reallocating/reindexing!
- **Dynamic Alignment Path (Reindex on the Fly)**: When `dates_hash` or `insts_hash` differs (e.g., loading a dataset from another vendor with a different instrument subset, regional universe, or extended date history, or when a researcher subsets the backtest universe in `info.json`), `DataManager` reads the embedded `"dates"` and `"instruments"` arrays from the JSON metadata block and automatically **reindexes** the 2D array onto the active backtest universe `(target_dates, target_instruments)`, filling missing cells with `NaN` (and optionally forward-filling low-frequency fundamental data).

---

## 3. Universe Definition Discovery

`DataManager` resolves the target backtest universe `(dates, instruments)` using the following precedence:

1. **`data/info.json` (or `data/universe.json`)**:
   - May specify inline `"dates": [...]` and `"instruments": [...]`, or paths `"dates_file"` and `"instruments_file"`, along with `"rows"`, `"columns"`, and `"target"`.
2. **`data/universe_dates.csv` & `data/universe_instruments.csv`** (or fallback `wrds/universe_dates.csv` & `wrds/universe_instruments.csv`):
   - Single-column newline-delimited lists of ISO dates (`YYYY-MM-DD`) and instrument identifiers (`ISIN` / `PERMNO` / ticker).

---

## 4. Usage Examples

### Python Data Engineering Pipeline (`python/comet/dmgr.py`)
```python
from comet.dmgr import DataManager

# Initialize with target universe
dm = DataManager(data_dir="data", wrds_dir="wrds", use_mmap=True)

# 1. Read any field (automatically prefers .bin over .csv.gz and aligns to universe)
close_df = dm.get_dataframe("close")          # Returns pandas.DataFrame (if pandas installed) or AlignedFrame
close_np = dm.get_numpy("close", align=True)  # Returns (T, N) numpy.ndarray or numpy.memmap

# 2. Write a pandas DataFrame or raw vendor table with arbitrary dates/columns into .bin
#    (Can save either pre-aligned to target universe, or with its own native coordinates!)
dm.write_bin("custom_alpha", df_or_array, dates=custom_dates, instruments=custom_insts, dtype="f64")

# 3. Batch convert existing data/*.csv.gz to data/*.bin
dm.convert_all_csv_gz_to_bin()
```

### Rust Backtest Runtime (`crates/runtime/src/dmgr.rs`)
```rust
use runtime::dmgr::{DataManager, LoadMode};

// Default LoadMode::Mmap maps .bin files into virtual memory and aligns to the active universe
let mut dmgr = DataManager::with_mode("./data".into(), None, LoadMode::Mmap);
let close_tensor = dmgr.get_data("close").unwrap();
```
