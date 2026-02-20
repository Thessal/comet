# Remaining Operators to Translate & LLVM Types

## Operator Types Table
The following table declares the LLVM types of each operator. Note that in C-ABI, functions typically take output arrays as parameter pointers (`f64*` -> LLVM `ptr`) rather than returning arrays directly. Ambiguous types (like boolean or string maps) are marked to be resolved later.

| Operator Classification | Operators | Parameters | Output Array | Notes |
|---|---|---|---|---|
| **Data Source** | `data` | - | - | *Ambiguous*: String ID to data. Skipped for now. |
| **Generators** | `const` | `value (f64)` | `out (ptr)` | |
| **Unary** | `abs` | `signal (ptr)` | `out (ptr)` | |
| **Stateful Unary** | `ts_delay`, `ts_diff`, `ts_mean`, `ts_sum`, `ts_decay_linear`, `ts_decay_exp`, `ts_std`, `ts_mae`, `ts_min`, `ts_max`, `ts_argmin`, `ts_argmax`, `ts_argminmax`, `ts_ffill` | `state (ptr)`, `signal (ptr)`, `period (i64)`, `len (i64)` | `out (ptr)` | `period` and `len` might be passed via `init` or `step`. |
| **Cross Sectional (Completed)** | `cs_rank`, `cs_zscore` | `state (ptr)`, `signal (ptr)`, `len (i64)` | `out (ptr)` | |
| **Binary** | `add`, `mid`, `subtract`, `divide`, `multiply`, `equals`, `greater`, `less`, `min`, `max` | `state (ptr)`, `x (ptr)`, `y (ptr)`, `len (i64)` | `out (ptr)` | |
| **Modifiers** | `clip`, `tail_to_nan` | `state (ptr)`, `signal (ptr)`, `lower (f64)`, `upper (f64)` | `out (ptr)` | |
| **Complex Logic** | `tradewhen` | `state (ptr)`, `signal (ptr)`, `enter (ptr)`, `exit (ptr)`, `period (i64)` | `out (ptr)` | |
| **Conditional** | `where` | `state (ptr)`, `condition (ptr)`, `val_true (ptr)`, `val_false (ptr)` | `out (ptr)` | `condition` is logically implicit bool, but mapped to `f64` in float-space. |
| **Matrix Operations** | `covariance` | `state (ptr)`, `returns (ptr)`, `lookback (i64)` | `out (ptr)` | Needs 3D Output? Or flattened matrix. |

## Remaining Operator Backlog
The following operators from `operators-example.py` still need to be translated to Rust LLVM C-ABI:

- `data`
- `const`
- `abs`
- `ts_delay`
- `ts_diff`
- `ts_sum`
- `ts_decay_linear`
- `ts_decay_exp`
- `ts_std`
- `ts_mae`
- `ts_min`
- `ts_max`
- `ts_argmin`
- `ts_argmax`
- `ts_argminmax`
- `ts_ffill`
- `tradewhen`
- `where`
- `mid`
- `subtract`
- `divide`
- `multiply`
- `equals`
- `greater`
- `less`
- `min`
- `max`
- `clip`
- `tail_to_nan`
- `covariance`
