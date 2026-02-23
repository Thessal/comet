# Compilation Strategy (`compile.md`)

This document defines the compilation strategy for Comet.

## 1. Targets

We evaluated three options:
1.  **List of Expressions (Python Runtime)**: Easy but slow. Valid for prototyping.
2.  **Bytecode (VM)**: Good for portability, but complex to implement a performant VM for combinatorial logic.
3.  **LLVM IR Generation**: **SELECTED STRATEGY**.

## 2. LLVM IR Generation

The main Comet compiler codebase (the "parser") acts as a generator of LLVM IR (or bitcode).

### Workflow
1.  **Source**: `example.cm`
2.  **Comet Compiler**:
    -   Parses and resolves Type/Behavior/Function logic.
    -   Expands the `flow` into a "Product Space" of concrete execution trees, finding all compatible combinations.
    -   Prunes invalid trees using Semantic Properties.
3.  **Codegen**:
    -   For each valid Tree in the Product Space, generate corresponding LLVM IR (or bitcode).
    -   Unique variant_id is assigned for each LLVM IRs. 
    -   (We don't need it currently, but in the future, memoization can be used to optimize computation tree and reduce recalculation of same subtrees among valid Trees.)
4.  **Compilation (`llvm` / linker)**:
    -   Compiles and links the generated LLVM code with the `stdlib` dynamic library (`.so`) into a high-performance executable strategy.

## 3. Generated Library API

The generated Rust library exposes a C-compatible (or Python-compatible via PyO3) interface to drive the strategies.

### 3.1 Metadata & Tags

The library exports a queryable structure for metadata.

```rust
pub struct StrategyMeta {
    pub variant_id: u32,
    pub tags: HashMap<String, String>, 
    // Keys: "lookback", "author", "generation_date", "version", "universe", "instrument_type", "used data", ...
    // TODO: Expression of the strategy that can be used for the symbolic regression.
    // IMPORTANT - We have to do causal analyais to find factors that results good alpha.
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
-   NOTE: we need to think about how to do the memory management. There are states for each operator, so passing it as a single blob might be tricky. Maybe we can use a hashmap to store the states of each operator.
-   State is a ring buffer that is passed between calls to `generate`. There are two types of states:
    -   Fixed size : Flattened, fixed size matrix. data type is f64 and it is flattened into a 1D array.
    -   Dynamic size : Iliffe vector. (not implemented yet). Can be used for strings or sparse graphs. (low priority)
-   **Purpose**: Live trading / Incremental update.
-   **Input**:
    -   `variant_id`: Strategy logic.
    -   `new_data`: Just the new incoming bars.
    -   `state`: The serialized state from the previous step.
-   **Output**:
    -   `Signal`: Signal for the new data points.
    -   `State`: Updated serialized state.

### Why LLVM?
-   **Type Safety**: The LLVM IR generator enforces valid memory access and strict type passing with the `stdlib` components.
-   **Performance**: Combinatorial explosion requires native speed. LLVM's advanced optimization passes generate highly efficient machine code.
-   **Compatibility**: Direct linking with the Rust-generated `stdlib` `.so` library allows reusing complex stateful kernels easily without incurring FFI overhead during dynamic execution.
