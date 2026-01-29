# Synthesis & Property Propagation

The Comet synthesizer transforms a semantic flow definition into an execution graph, resolving types, selecting implementations, and verifying constraints.

## 1. Property Propagation system

Properties (e.g., `NonZero`, `Ranged`, `Masked`) are metadata attached to types or variables in the symbol table.

### 1.1 Source Generation
Variables derived from `type` definitions inherit properties from the logical type.
```comet
type Volume : NonZero ...
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
