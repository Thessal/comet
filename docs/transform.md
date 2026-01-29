# Transformation & Synthesis Logic (`transform.md`)

This document defines the **Semantic Synthesis** process, which transforms a functional `Flow` into a set of combinatorial execution trees.

## 1. The Synthesis Loop

The compiler iterates through the function definitions, performing **Type Class Instance Resolution**.

### Step 1: Generators / Sources
When checking `x = universe Earnings`:
1.  Look up the `Earnings` type.
2.  Load all associated Type Class Instances (`NonZero`, `Monetary`, etc.).

### Step 2: Function Resolution (`compare a b`)
When encountering `ratio = compare vol1 vol2`:
1.  Identify the Type Class: `Comparator`.
2.  **Instance Search**: Find ALL instances of `Comparator` matching the types of `vol1` and `vol2`.
    *   *Match 1*: `Ratio` (Requires `NonZero vol2`)
    *   *Match 2*: `Spread` (Requires `SameUnit vol1 vol2`)
3.  **Constraint Check**: Verify constraints against the known instances of variables.
    *   If `vol2` has `instance NonZero`, then `Ratio` is valid.
    *   If `vol1` and `vol2` share unit tag, then `Spread` is valid.
4.  **Branching**: Create a separate execution path (Tree Node) for each passed instance.

### Step 3: Filtering (Constraints)
Explicit constraints in valid function signatures (`| Condition a`) act as pruners.
If a branch allows a datatype that violates a downstream constraint, that entire branch is pruned eagerly.

## 2. Output Generation
The result is a forest of "Expression Trees", where each node is a specific, concrete Instance implementation (e.g., `Ratio_Impl` call).

These trees are passed to the Code Generator to emit Rust code.
