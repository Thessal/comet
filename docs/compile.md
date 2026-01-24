# Compilation Strategy (`compile.md`)

This document defines the compilation strategy for Comet.

## 1. Targets

We evaluated three options:
1.  **List of Expressions (Python Runtime)**: Easy but slow. Valid for prototyping.
2.  **Bytecode (VM)**: Good for portability, but complex to implement a performant VM for combinatorial logic.
3.  **Rust Source (Transpilation)**: **SELECTED STRATEGY**.

## 2. Transpilation to Rust

Comet acts as a **Generator** of Rust code.

### Workflow
1.  **Source**: `example.cm`
2.  **Comet Compiler**:
    -   Parses and resolves Type/Trait/Imple logic.
    -   Expands the `Flow` into a "Product Space" of concrete execution trees.
    -   Prunes invalid trees using Semantic Properties.
3.  **Codegen**:
    -   For each valid Tree in the Product Space, generate a unique Rust struct/function (e.g., `Strategy_Variant_142`).
    -   Generate a `lib.rs` that exposes the standard API.
4.  **Rust Compiler (`cargo`)**:
    -   Compiles the generated Rust code into a high-performance library (`.so` / `.dll` / `.rlib`).

## 3. Generated Library API

The generated Rust library exposes a C-compatible (or Python-compatible via PyO3) interface to drive the strategies.

### 3.1 Metadata & Tags

The library exports a queryable structure for metadata.

```rust
pub struct StrategyMeta {
    pub variant_id: u32,
    pub tags: HashMap<String, String>, 
    // Keys: "lookback", "author", "generation_date", "version", "universe", "instrument_type"
}
```

### 3.2 Functions

The interface is stateless at the library level; state is passed in/out.

#### `select(variant_id: u32) -> DataSpec`
-   **Purpose**: Tells the caller what data is needed for this specific combination.
-   **Input**: Index of the strategy variant.
-   **Output**: Description of required inputs (instruments, lookback windows, etc.).

#### `generate_all(variant_id: u32, history: DataFrame) -> (Signal, State)`
-   **Purpose**: Cold start / Backtest. Runs the strategy over the entire history.
-   **Input**: 
    -   `variant_id`: Which strategy logic to use.
    -   `history`: The full dataset (pandas/polars DataFrame).
-   **Output**:
    -   `Signal`: The resulting time-series of signals.
    -   `State`: Serialized byte blob representing internal state (e.g., rolling window buffers) at the end of history.

#### `generate(variant_id: u32, new_data: DataFrame, state: State) -> (Signal, State)`
-   **Purpose**: Live trading / Incremental update.
-   **Input**:
    -   `variant_id`: Strategy logic.
    -   `new_data`: Just the new incoming bars.
    -   `state`: The serialized state from the previous step.
-   **Output**:
    -   `Signal`: Signal for the new data points.
    -   `State`: Updated serialized state.

### Why Rust?
-   **Type Safety**: The generated code is statically checked by `rustc`.
-   **Performance**: Combinatorial explosion requires native speed.
-   **Parallelism**: Rust's `Rayon` or `Tokio` can easily run independent strategy variants in parallel.
