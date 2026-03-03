# DAG-Based Optimization & Variant Sampling

This plan outlines the optimization strategy for AST synthesis and compilation, replacing exhaustive Rust code generation with targeted, multi-stage sampling and Graph-based CSE (Common Subexpression Elimination).

## 1. Decoupling Synthesis from Codegen & Multi-Stage Sampling
As discussed, synthesizing exhaustive permutations of `AST`s in Rust is extremely fast and harmless. The bottleneck is strictly the Rust code Generation phase (`codegen`), which suffers when processing tens of thousands of variants into a single `.so` module.

We will keep the Synthesizer completely exhaustive, but we will introduce a filtering and sampling step immediately before Codegen.

**The CLI Design:**
Instead of selecting explicit variant IDs (which are arbitrary and unstable), we will use statistical sampling at the command line:
```bash
cargo run -- ./50kcombination.cm --sample-rate=0.01 --exclusive-sample-stages=100
```
- **`--sample-rate=0.01`**: Each stage will select exactly 1% of the total exhaustively generated variants. For 50k total variants, this is 500 variants.
- **`--exclusive-sample-stages=100`**: The compiler run will emit 100 separate `.so` files.
- **Exclusive Sampling**: Crucially, the sampler will pop elements from the synthesized list *without replacement* across stages. 
  - Stage 1 selects 500 random elements and generates `output_stage_1.so`. 
  - Stage 2 selects 500 elements from the remaining 49,500 and generates `output_stage_2.so`, and so on.
  This allows parallel, distributed backtesting across multiple `.so` artifacts without duplicating work between buckets.

## 2. Contextual Weighting & The Evolutionary "Gene"
Assigning a static `weight=0.8` to an `Fn` is flawed because an `Fn`'s importance is highly contextual to the target `Behavior`. We will attach probability matrices directly to the `Behavior` definition context.

**Defining the "Gene":**
To support future evolutionary loops (Genetic Algorithms / Bayesian Optimization), the "Gene" object must be explicit, parsable, mutatable, and pruneable. 
- A **Gene** is the matrix of function selection probabilities mapped to a specific `Behavior`.
```json
{
  "behavior": "Indicator",
  "gene": {
    "functions": {
      "Diff": 0.7,
      "MACD": 0.6,
      "Rank": 0.1
    },
    "default_weight": 0.05
  }
}
```
- **Mutation**: An external optimizer can tweak values (e.g., multiply `Diff` by 1.1) or introduce new functions into the dictionary.
- **Pruning**: If an ML model detects that [Rank](file:///home/jongkook90/antigravity/comet/src/stdlib/cs_rank.rs#5-6) is returning zero alpha across all samples, its gene weight is set to `0.0`, entirely pruning that branch during the Synthesis phase.
- **Hybridization**: Two successful genes for `Indicator` could be crossover-averaged to create the next generation's search parameters. 

This JSON/Matrix representation keeps the genetic abstraction entirely separate from the static `Fn` definitions in the user's `.cm` files.

## 3. DAG Construction via `Display` (Compile-Time CSE)
Regardless of how the variants are selected for a specific `.so` stage, we must eliminate redundant computations shared across the sampled set.

1. **Hash-Consing Pass:** Input the selected list of $N$ AST root nodes (e.g., 500 nodes).
2. Initialize a `HashMap<String, NodeId>` and an array `Vec<DagNode>`.
3. Iterate through each of the $N$ sampled ASTs from the bottom up (post-order traversal).
4. **The `StructuralHash`:** We calculate the identity of an AST node by utilizing the `Display` trait. When formatted, `ts_mean(data("volume"), 5)` explicitly represents a unique string. This formatting inherently captures the operation and the arguments. 
   - If the `Display` string exists in the HashMap, we reuse the existing `NodeId`.
   - If it doesn't, we insert the `Display` string into the HashMap, map it to a new `NodeId`, and push the node onto the `DagNode` array. This guarantees zero collisions for distinct operations.

## 4. Rust Code Generation & Threading Considerations
By mapping the AST structures through the DAG array, iterating `0..len` provides a Topologically Sorted execution order.
Operator states are managed via a `PipelineState` struct.

**Threading the Execution DAG:**
Because our end goal is to compile these static formulas into portable native Rust dynamic libraries (`strat.so`).
Therefore, **the generated Rust code will run as a single-threaded serial execution loop.** 

While the nodes execute serially within a single variant's `execute` function, we achieve parallelization horizontally: The Python/Frontend architecture can deploy and execute 100 independent `.so` researches across 100 different worker processes in parallel. By combining memory-optimal, highly compressed `.so` files (thanks to CSE) with wide horizontal multi-processing, system throughput scales securely and predictably.
