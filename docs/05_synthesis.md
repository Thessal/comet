# Synthesis 

The Comet synthesizer expands a semantic definition (Flow, Behavior) in the "AST", which results iteration over a set of function(Fn) trees.

The synthesizer checks types, and constraints (Type, Category).

This is done by iterating over the cartesian product space of Behavior and Flow.


## Behavior expansion
- Each behavior is expanded into a list of implementations.
- Implementation is a tree of `Fn` calls.

- A valid implementation is a tree of `Fn` calls, that satisfies the `Behavior`'s constraints.
  - It must consume arguments of the behavior **exactly once**, as function inputs.
  - Behavior arguments must be compatible to the function inputs constraints, so it is able to be used as the function inputs.
  - It must return a value that satisfies the return constraint of the behavior.

- For example, `behavior Comparator(signal: DataFrame, eps: Float Nonzero Optional, reference: DataFrame) -> DataFrame Indicator`
   - One possible behavior call is `Comparator(signal=volume, reference=adv20, eps=0.1, depth=2)`
      - Literal `0.1` validates call constraints, so it is user's responsibility to provide a valid literal.
   - Assume that there are functions:
      - `Fn Consume(b: Float Optional) -> ()`
      - `Fn Rank(a: DataFrame) -> Normalized DataFrame`
      - `Fn RankNonzero(a: DataFrame, eps: Float Nonzero) -> DataFrame Normalized Nonzero`
      - `Fn Diff(a: DataFrame, b:DataFrame) -> DataFrame Indicator`
      - `Fn Divide(a: DataFrame, b: DataFrame Nonzero) -> DataFrame Indicator`
      - `Fn OtherOperation(a: DataFrame, b: DataFrame) -> DataFrame`
   - This behavior call is expanded into a full exhaustive set of valid instances:
      - **Depth 1 implementations** (No nested function calls, only root binary operations):
         - `Consume(eps=eps); Diff(volume, adv20)`
         - `Consume(eps=eps); Diff(adv20, volume)`
      - **Depth 2 implementations** (Max nested depth is 1 for arguments):
         - *Using `Divide`, which requires `RankNonzero` to consume `eps` for the denominator:*
            - `Divide(volume, RankNonzero(adv20, eps=eps))`
            - `Divide(adv20, RankNonzero(volume, eps=eps))`
            - `Divide(Rank(volume), RankNonzero(adv20, eps=eps))`
            - `Divide(Rank(adv20), RankNonzero(volume, eps=eps))`
         - *Using `Diff`, explicitly consuming `eps` inside `RankNonzero` for one argument:*
            - `Diff(volume, RankNonzero(adv20, eps=eps))`
            - `Diff(RankNonzero(adv20, eps=eps), volume)`
            - `Diff(adv20, RankNonzero(volume, eps=eps))`
            - `Diff(RankNonzero(volume, eps=eps), adv20)`
            - `Diff(Rank(volume), RankNonzero(adv20, eps=eps))`
            - `Diff(RankNonzero(adv20, eps=eps), Rank(volume))`
            - `Diff(Rank(adv20), RankNonzero(volume, eps=eps))`
            - `Diff(RankNonzero(volume, eps=eps), Rank(adv20))`
         - *Using `Diff`, consuming `eps` globally via `Consume`:*
            - `Consume(eps=eps); Diff(Rank(volume), adv20)`
            - `Consume(eps=eps); Diff(volume, Rank(adv20))`
            - `Consume(eps=eps); Diff(Rank(adv20), volume)`
            - `Consume(eps=eps); Diff(adv20, Rank(volume))`
            - `Consume(eps=eps); Diff(Rank(volume), Rank(adv20))`
            - `Consume(eps=eps); Diff(Rank(adv20), Rank(volume))`
   - Invalid instances:
      - `Diff(volume, adv20)` (eps is not consumed)
      - `Divide(RankNonzero(volume, eps=eps), RankNonzero(adv20, eps=eps))` (eps is consumed twice)
      - `Divide(Rank(Rank(volume)), RankNonzero(adv20, eps=eps))` (depth is 3, but max depth is 2)
      - `OtherOperation(Rank(volume), RankNonzero(adv20, eps=eps))` (Output is not `Indicator`)

- Search space can be refined further.
   - If `Diff` and `Divide` requires same category for the input,
      - `Fn Diff(a: DataFrame 'a, b:DataFrame 'a) -> DataFrame 'a Indicator`
      - `Fn Divide(a: DataFrame 'a, b: DataFrame 'a Nonzero) -> DataFrame 'a Indicator`
   - **Category Capture Unification**: A constraint variable like `'a` captures the *exact* set of unspecified categories from the runtime argument it binds to. When `'a` is shared across multiple parameters (e.g., `a: DataFrame 'a, b: DataFrame 'a`), it enforces **strict category unification**: all arguments sharing `'a` must hold the identical captured category set, effectively pruning asymmetric branches.
   - Then the search space reduces into
      - **Depth 1 implementations** (No nested function calls):
         - `Consume(eps=eps); Diff(volume, adv20)` (Both are identically `DataFrame`)
         - `Consume(eps=eps); Diff(adv20, volume)` (Both are identically `DataFrame`)
         - *Note:* `Divide` requires `Nonzero` on `b`, so it cannot be satisfied natively by raw inputs at Depth 1.
      - **Depth 2 implementations** (Max nested depth is 1 for arguments):
            - `Divide(Rank(volume), RankNonzero(adv20, eps=eps))` (Normalized, Normalized Nonzero)
            - `Divide(Rank(adv20), RankNonzero(volume, eps=eps))` (Normalized, Normalized Nonzero)
            - `Consume(eps=eps); Diff(Rank(volume), Rank(adv20))` (Normalized, Normalized)
            - `Consume(eps=eps); Diff(Rank(adv20), Rank(volume))` (Normalized, Normalized)

- Output of the behavior call is the output of the function call, but the constraints acquired by the output is the constraint of the behavior call, which is only subset of the direct function output.

## Algorithm for Exhaustive Search

A bottom-up dynamic programming (or iterative deepening) approach is well-suited for this exhaustive search. Since a valid implementation can consist of a primary expression tree alongside multiple independent side-effect trees (like diverse `Consume` functions that return `()`), the algorithm first generates all valid subtrees and then assembles them into valid forests.

**Step 1: Generate All Subtrees**
We track a pool of valid subtrees `P`. Each subtree state has:
- `tree`: The AST of the function calls (e.g., `Rank(volume)`).
- `consumed_args`: A set of behavior argument names that this subtree uses.
- `output_constraint`: The resulting resolved constraint of this subtree.
- `depth`: The max nested depth of this subtree.

1. **Initialization (Depth 0):**
   - For each argument `arg` in the behavior call, add a base state to `P`: 
     `tree = arg.name`, `consumed_args = {arg.name}`, `output_constraint = arg.constraint`, `depth = 0`.

2. **Iterative Search (Depth 1 to `max_depth`):**
   - For `d = 1` to `max_depth`:
     - For each function `F` in the `Fn` library with `N` arguments:
       - Find all `N`-tuples of subtrees `(S_1, S_2, ..., S_N)` from `P` such that:
         - At least one `S_i` has `depth = d - 1` (to ensure exactly reaching depth `d`).
         - The sets of `consumed_args` among all `S_i` are mutually disjoint (no argument is consumed twice).
         - The `output_constraint` of each `S_i` satisfies `F`'s corresponding argument constraint.
         - Constraint variables (like `'a`) unify correctly across all `S_i`.
       - For each valid tuple, add the new state to `P`:
         - `tree = F(S_1.tree, ..., S_N.tree)`
         - `consumed_args = Union(S_i.consumed_args)`
         - `output_constraint = Resolved return constraint of F`
         - `depth = d`

**Step 2: Assemble Valid Implementations (Forests)**
A complete implementation is a subset of mutually disjoint subtrees that perfectly consume all arguments and return the correct behavior constraint.

1. **Forest Assembly:**
   - Find all subsets of subtrees from the pool `P` such that:
     1. **Return Constraint:** Exactly one subtree $T_{main}$ produces an `output_constraint` that satisfies the behavior's expected `return_constraint`.
     2. **Side Effects:** All other subtrees $T_{side}$ in the subset must return `()` (e.g., diverse side-effect functions like `ConsumeInt`, `ConsumeFloat`, `LogData`, etc.).
     3. **Exact Consumption:** The `consumed_args` between all subtrees in the subset are mutually disjoint, and their union perfectly exactly equals the set of all behavior arguments.
2. **Yielding:**
   - Return these mutually valid subsets as the exhaustive list of implementations. (e.g., `Consume(eps=eps); Diff(volume, adv20)` is simply natively assembled and recognized as the subset `{ Consume(eps), Diff(volume, adv20) }`).


## Iteration over Flow

Flow is a cartesian product space of behaviors and sets of literals.

- Integer literals expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Int) -> DataFrame 
Flow historical_volume {
   variousdays = [5, 21, 252] # choice(5, 21, 252)
   // or variousdays = [10..10..50] # range(10, 50, step=10).
   ts_mean(a=data(id="volume"), window=variousdays)
}
```
- Floating point literal expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Float) -> DataFrame 
Flow historical_volume {
   variousdays = [5.0, 21.0, 252.0] # choice(5.0, 21.0, 252.0)
   // or variousdays = [10.0..10.0..50.0] # range(10.0, 50.0, step=10.0).
   ts_mean(a=data(id="volume"), window=variousdays)
}
```

- String literal expansion example : The following code yields three implementations.
```comet
Fn data(id: String) -> DataFrame
Fn ts_mean (a: DataFrame, window: Int) -> DataFrame 
Flow historical_volume {
   variousdata = ["volume", "adv20"] # choice("volume", "adv20")
   ts_mean(a=data(id=variousdata), window=21)
}
```
