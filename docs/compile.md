# Compilation Strategy (`compile.md`)

This document defines the compilation strategy for Comet.

## 1. Targets

We utilize **Rust Source Transpilation**. Comet acts as a Generator of Rust code.

## 2. Transpilation to Rust

### Workflow
1.  **Source**: `example.cm` (Clean-like syntax)
2.  **Comet Compiler**: Parses, Resolves Types/Instances, Synthesizes Product Space of Execution Trees.
3.  **Codegen**: Generates Rust structs/functions for each valid Tree.
4.  **Rust Compiler**: Compiles to library.

## 3. Generated Library API

The generated library exposes a stateful "step" interface for incremental execution.

### 3.1 Metadata
```rust
pub struct StrategyMeta {
    pub variant_id: u32,
    pub logic_path: Vec<String>,
}
```

### 3.2 Execution Interface

The core contract is that the library is stateless logic, but it handles state transition.

```rust
/// Step function for incremental execution
/// 
/// # Arguments
/// * `variant_id` - IDs the specific synthesized strategy variant
/// * `old_state` - Serialized state from previous step (or empty for cold start)
/// * `new_data` - The new increment of data (e.g. 1 new bar, or a chunk)
/// 
/// # Returns
/// * `new_state` - Updated state (to be passed to next call)
/// * `result` - The computed signal for this step
pub fn step(
    variant_id: u32, 
    old_state: &[u8], 
    new_data: &DataFrame
) -> (Vec<u8>, DataFrame)
```

### Why Rust?
Top-tier performance and type safety are critical when managing the combinatorial explosion of synthesized strategies.
