<!-- 
Synthesizer Logical Flaws & Missing Information Identified:

1. **Constraint Variable Resolution (Unification)**: 
   The document dictates you can use constraint variables like `'a` to link inputs and outputs (e.g., `signal: DataFrame 'a -> DataFrame 'a`). However, it doesn't specify how exactly `'a` is bound when the input has multiple properties (e.g. `DataFrame Monetary Indicator`). Is `'a` the entirety of the extra properties `{Monetary, Indicator}`? Furthermore, if multiple arguments use `'a`, do their incoming property sets have to match intimately, or do we take the union/intersection?

2. **Property Pass-through for standard `Fn`s**: 
   Standard library functions (`Fn`s) cleanly assert a specific output (e.g., `Fn Ratio(...) -> DataFrame Finite`). If a flow passes an input that happened to be `DataFrame Volume` into `Ratio`, does the resulting output completely lose the `Volume` property? If functions stubbornly clamp to their explicitly declared return constraint, the graph will uncontrollably bleed specific semantic properties needed by subsequent behaviors. Can `Fn`s specify pass-throughs?

3. **Flow Input Parameters**: 
   The spec suggests `Flow <name> { ... } -> <constraint>`, yet the Flow definitions deliberately lack explicit input arguments in the syntax. Yet they are called in functions like `volume_spike(...)`. If flows can be passed inputs externally, how are arguments bound inside the block without declaring a `(signal: Constraint, ...)` arg-list?

4. **Cartesian Explosion**: 
   If statement A matches 3 `Fn` implementations, and statement B matches 4, the Flow now has 12 combinations. Deep pipelines will quickly experience factorial combinations. Is there a culling step missing?
   
5. **AST vs "Real AST"**: 
   "Real AST" isn't concretely specified. I am formalizing it as `RealProgram` containing resolved instances of `RealFlowCombinations`, where ALL behavior mappings (`Expr::Call`) have been completely expanded into direct `Fn` calls (`RealExpr::CallFn`).
-->

# Synthesis & Property Propagation

The Comet synthesizer transforms a semantic flow definition into an execution graph, resolving types, selecting implementations, and verifying constraints.

## 1. Property Propagation system

Properties (e.g., `NonZero`, `Ranged`, `Masked`) are metadata attached to types or variables in the symbol table.

### 1.1 Source Generation
Variables derived from `type` definitions inherit properties from the logical type.
```comet
flow min_vol = Volume() // min_vol has { NonZero }
```

### 1.2 Strict Pruning
If a function or behavior call has no valid implementations (e.g. all candidates fail constraints), the synthesis branch is **pruned**. The synthesizer does not fallback to identity or recursion for failed ops.


## 2. Constraint Checking

Type constraints in Behavior and Function arguments are verified against the properties of the inputs during synthesis.
*Example*: `fn apply_filter(data: Ranged)`.
If the input argument does not have the `Ranged` property, the synthesis branch is rejected.

## 3. Multiple Function Expansion

The synthesizer explores all valid function implementations of a Behavior.
*Example*: `Comparator` might have `Ratio` and `RankDiff`. Both are explored (context branching) if their constraints are satisfied.
```comet
fn Ratio(dividend: DataFrame, divisor: DataFrame NonZero)
```
If `divisor` lacks `NonZero`, `Ratio` variant is dropped.
