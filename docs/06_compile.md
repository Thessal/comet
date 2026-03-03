## 1. Targets

Rust code is generated from the set of function trees by the compiler.

## 2. Rust code Generation

The main Comet compiler codebase (the "parser") acts as a generator of Rust code (or bitcode).

### Workflow
1.  **Source**: `example.cm`
2.  **Comet Compiler (Frontend -> Synthesis)**:
    -   Parses typed primitive literals (`String`, `Int`, `Float`, `Bool`) and explicitly binds categorical constraints and recursion `depth` limits.
    -   Performs validation on all `Flow` body expressions to verify that all called functions are defined in the globally available `known_functions` Set.
    -   Extracts nested properties and resolves `Flow` variables by running tree transformations using `substitute_expr` before synthesis.
    -   Translates the abstract AST to a concrete `RealAST` by deploying a dynamic programming Exhaustive Search Algorithm.
    -   Expands Behaviors into a "Product Space" of valid implementations modeled as disjoint `Fn` Forests where exactly one tree matches the return constraint and all side-effect trees return `()`. see examples/consume_minimal.cm
    -   Enforces **Category Capture Unification** (`'a`) to strictly prune asymmetric functional branches during constraint matching.  TODO: write examples
    -   Expands Behaviors into a product space of valid implementations modeled as disjoint `Fn` Forests.
3.  **Codegen (RealAST -> Rust code)**:
    -   For each valid Forest in the product space, generate corresponding Rust code strings.
    -   The Forest is sampled and merged into a single execution graph. (DAG)
4.  **Compilation (`cargo` / linker)**:
    -   Compiles and links the generated Rust code with the `stdlib` dynamic library (`.so`) into a high-performance executable strategy.


## 3. Compiler & Rust Codegen Specifications

1. **Primitive Literal Constraints**:
    - When `RealAST` evaluates native literals (`Int`, `Float`), these should be injected into the Rust code as `constant` declarations explicitly compiled inside the variant's function execution block. 
    - `String` literals (like `data("volume")`) must be saved in `StrategyMeta` tags. We are not sure how to use them yet, so its implementation is low priority.

2. **Category Property Rust Tags (Type Stripping)**:
    - `CategoryExpr` boundaries (`'a`, `Normalized`, etc.) exist STRICTLY at compile-time to prune invalid Cartesian `Fn` forests. 
    - Once `synthesis.rs` resolves a valid `RealAST`, the Rust generator drops all Semantic Categories. Rust exclusively recognizes base structural shapes (e.g. `DataFrame` -> `*double` arrays, `Int` -> `i64`).

3. **Execution Ring Buffer State Definition**:
    - The compiled `.so` C-ABI must explicitly define the passed `State`.
    - Passing state as a type-erased pointer: The state blob `*u8` passed by the caller should map directly into a static ring buffer struct.
    - The Rust code will allocate offsets dynamically across the active `CallFn` nodes (e.g., `ts_mean` requires an `f64` rolling accumulator offset; `Diff` requires none). Thus, `State` sizes are deterministically precomputed during Rust Generation.

## 4. Generated Library API

The generated Rust library exposes a C-compatible (or Python-compatible via PyO3) interface to drive the strategies.

### 4.1 Metadata & Tags

The library exports a queryable structure for metadata.

// subject to change
```rust
pub struct StrategyMeta {
    pub variant_id: u32,
    pub tags: HashMap<String, String>, 
    // Keys: "lookback", "author", "generation_date", "version", "universe", "instrument_type", "used data", ...
    // TODO: Expression of the strategy that can be used for the symbolic regression.
    // IMPORTANT - We have to do causal analysis to find factors that results good alpha.
}
```

### 4.2 Functions
// subject to change

The interface is stateless at the library level; state is passed in/out.
<!-- TODO: We need to define the DataSpec and implement the select function. -->
<!-- 
#### `select(variant_id: u32) -> DataSpec`
-   **Purpose**: Tells the caller what data is needed for this specific combination.
-   **Input**: Index of the strategy variant.
-   **Output**: Description of required inputs (instruments, lookback windows, etc.). -->

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
