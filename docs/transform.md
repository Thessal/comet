# Transformation & Synthesis Logic (`transform.md`)

This document defines the **Semantic Synthesis** process, which transforms a `Flow` into a set of combinatorial execution trees.

## 1. The Synthesis Loop

The compiler iterates through the `Flow` statements, maintaining a **Combinatorial State**.

**State**: `List<Context>`
Where `Context` is a map of `Variable -> Value`.

12: ### Step 1: Generators / Data
13: 
14: When checking `data x -> Universe(Earnings)`, the compiler:
15: 1.  Queries the Symbol Table for all Types deriving `Earnings`.
16: 2.  Creates a new Context branch for each match.
17:     -   Context 1: `{ x: EBIT (USD, NonZero) }`
18:     -   Context 2: `{ x: EBITDA (USD, NonZero) }`

### Step 2: Behavior Resolution (`Comparator(x, y)`)

When encountering `spike = Comparator(x, y)`:
1.  For each active Context:
    -   Resolve the types of `x` and `y`.
    -   Query the Symbol Table for all `fn` definitions of `Comparator` matching these types.
    -   **Constraint Check**: Evaluate the `where` clause (or type constraints) of each Function.
        -   Evaluation is done against the **Semantic Properties** of the types in the current Context.
2.  Branch the Context for each valid Function.
    -   If `Context 1` matches both "Ratio" and "Spread", it splits into `Context 1.A (Ratio)` and `Context 1.B (Spread)`.
    -   If `Context 1` matches none, this branch is **Pruned** (dead end).

Explicit `where` clauses in the Flow act as filters.
-   Evaluate the condition against the current Context.
-   If `False`, prune the branch.

## 2. Output Generation

At the end of the Flow, the compiler has a list of valid Contexts. Each Context represents a fully resolved "Call Graph" or "Expression Tree".

These trees are then passed to the Code Generator (Python/Rust backend) to emit the actual source code.
