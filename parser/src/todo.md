# Parser / Runtime Workflow Implications

By refactoring the `Runtime` to execute Polish Notation Sequences (`&[String]`) natively via a stack-machine instead of building a DAG graph, several compiler pipelines are broken and need to be addressed in the future:

1. **AST to DAG translation is Obsolete**: The `codegen` crate previously acted as an intermediary to build `DagBuilder` graphs from `Expr` ASTs. Since the runtime natively processes standard postfix/prefix combinations (like RL `SearchState`), the `Runtime::evaluate(node_id)` graph solver is removed.
2. **Standard Scripts Execution**: If a user runs `Examples/behavior_2.cm` utilizing standard language features (conditionals, loops, etc), compiling to a flat `Vec<String>` sequence might lose topological structural context like scopes or jumps.
3. **Caching**: Global sub-tree hashing (hash consing) that evaluated identical `f64` vectors on duplicated `node_id`s in the `LruCache` is no longer active. The `Runtime` completely calculates expressions sequentially, trading memory-efficiency caching for absolute pure native speed. This might impact performance on deeply nested repetitive loops (if any).

**TODOs for `parser` and `codegen`**:
- [ ] Implement robust `AST -> Sequence` emission inside `codegen` for the entire AST capability (moving beyond just math expressions).
- [ ] Remove `dag.rs` entirely from `codegen` once Sequence-based translation covers `Flow` blocks entirely.
