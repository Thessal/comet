# Synthesis & Property Propagation

The Comet synthesizer transforms a semantic flow definition into an execution graph, resolving types, selecting implementations, and verifying constraints.

## 1. Property Propagation system

Properties (e.g., `NonZero`, `Ranged`, `Masked`) are metadata attached to types or variables in the symbol table.

### 1.1 Source Generation
Variables derived from `Type` definitions inherit properties from the logical type.
```comet
Type Volume_Universe : Instrument derives { NonZero } ...
min_vol <- Volume_Universe // min_vol has { NonZero }
```

### 1.2 Strict Pruning
If a function or behavior call has no valid implementations (e.g. all candidates fail constraints), the synthesis branch is **pruned**. The synthesizer does not fallback to identity or recursion for failed ops.

### 1.3 Explicit Attachment (`ensures`)
Implementations and Functions can explicitly attach properties using the `ensures` clause.
```comet
Implementation ZScore ... ensures { Unbound, Ranged } { ... }
```
These properties are appended to the propagated properties.

## 2. Constraint Checking

`where` clauses on Functions and Implementations are verified against the properties of arguments during synthesis.
*Example*: `fn apply_filter(...) where data is Ranged`.
If the argument `data` does not have the `Ranged` property, the synthesis branch is rejected.

## 3. Multiple Implementation Expansion

The synthesizer explores all valid implementations of a Behavior.
*Example*: `Comparator` might have `Ratio` and `RankDiff`. Both are explored (context branching) if their constraints are satisfied.
```comet
Implementation Ratio ... where b is NonZero
```
If `b` lacks `NonZero`, `Ratio` variant is dropped.
