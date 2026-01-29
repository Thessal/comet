# FIXME / Implementation Status

## 1. Synthesis Engine (`src/comet/synthesis.rs`)
-   **Status**: Skeleton / Stub.
-   **Issue**: The current synthesizer does not perform actual Type Class Resolution. It identifies the entry point (`strategy`) but does not yet:
    -   Perform Unification of types to find matching Instances.
    -   Backtrack to explore valid implementation paths.
    -   Construct the `ExecutionGraph` from the resolved instances.
-   **Action**: Implement a standard Hindley-Milner style solver or a resolution engine (like logic programming) to satisfy `Constraints`.

## 2. Semantic Analysis (`src/comet/semantics.rs`)
-   **Status**: Basic Registration.
-   **Issue**: The analyzer registers declarations (`Adt`, `Class`, `Instance`, `Function`) into the `SymbolTable` but lacks:
    -   **Type Checking**: No verification that function bodies match their signatures.
    -   **Constraint Validation**: No check that `Instance` constraints are valid classes.
    -   **Cycle Detection**: No check for cyclic type definitions or superclass cycles.
-   **Action**: Implement a proper Type Checker pass.

## 3. Parser & Grammar (`src/comet/grammar.pest`, `src/comet/parser.rs`)
-   **Status**: Functional but fragile.
-   **Issue**:
    -   **Keywords**: The `keyword` exclusion list in `grammar.pest` is manual. New keywords must be added there to avoid being parsed as identifiers.
    -   **Precedence**: Expression parsing (`parse_logic`, `parse_term`, `parse_factor`) uses a simplified manual precedence climbing. It supports basic arithmetic/logic but may fail on complex mixed expressions or custom operators.
    -   **Type Refs**: parsing `a -> b -> c` is done via manual right-folding traversal which is simple but might miss edge cases in nested parentheses.
-   **Action**: Use `pest::pratt_parser` for robust expression parsing.

## 4. Code Generation (`src/comet/codegen.rs`)
-   **Status**: Interface Stub.
-   **Issue**: `codegen.rs` produces the Rust `step` function signature required by `compile.md`, but the body is currently a comment/placeholder because the Synthesis engine doesn't produce a fully populated `ExecutionGraph`.
-   **Action**: Connect the `ExecutionGraph` output from a working Synthesizer to the template emitter.

## 5. Standard Library (`comet_examples/stdlib.cm`)
-   **Status**: Parsing.
-   **Issue**: Contains "Marker Classes" (e.g., `class NonZero a`) which act as Semantic constraints. These are now supported by the parser, but the Synthesis engine doesn't yet know how to "prove" these properties (e.g. by data lineage analysis).
-   **Action**: Define the "Proof Strategy" for semantic properties (e.g., static analysis vs. runtime checks).
