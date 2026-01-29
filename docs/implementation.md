# Comet Compiler Implementation Plan (Quant DSL Version)

This document outlines the plan to build the Comet compiler, orchestrating the detailed design components.

**Reference Documents**:
-   [Lexical Spec](lex.md)
-   [Grammar Spec](parse.md)
-   [AST Spec](ast.md)
-   [Synthesis Logic](transform.md)
-   [Runtime Model](runtime.md)

## Architecture

### 1. Frontend (Lexing & Parsing)
-   **Goal**: Transform source text into the `Program` AST defined in `ast.md`.
-   **Tool**: `pest` (or `chumsky`).
-   **Validation**: Ensure syntax compliance with `parse.md`.

### 2. Semantic Analysis (Type Checking)
-   **Goal**: Validate the Logic and populate the Symbol Table.
-   **Process**:
    -   Register all `type` definitions and their constraints.
    -   Register all `behavior` signatures.
    -   Register all `fn` (Function) definitions.
    -   **Constraint Verification**: Validate that type constraints used in definitions are semantically valid (e.g., checking properties that exist).

### 3. Synthesis Engine (The Core)
-   **Goal**: Expand `Flow` definitions into concrete Execution Trees.
-   **Reference**: `transform.md`.
-   **Key Logic**:
    -   **Context Management**: Tracking active `Variable -> Value` bindings and their Semantic Properties.
    -   **Branching**: Splitting contexts when multiple Functions match a Behavior.
    -   **Pruning**: Dropping contexts where Constraints fail.
    -   **Built-in Logic**:
        -   **Functions**: Modular handlers (e.g., `src/comet/functions/divide.rs`) for complex logic like `divide(DataFrame, TimeSeries)`.

### 4. Backend (Code Generation)
-   **Goal**: Emit executable code for the Target Runtime.
-   **Strategy**: **Transpilation to Rust** (Selected Strategy).
-   **Reference**: `compile.md`, `runtime.md`.
-   **Process**:
    1.  **Expansion**: The Synthesis Engine produces a "Product Space" of valid execution trees.
    2.  **Codegen**: For each valid tree, generate a unique Rust struct/function (e.g., `Strategy_Variant_142`).
    3.  **Compilation**: Use `cargo` to compile the generated Rust code into a high-performance library (`.so` / `.dll`).
-   **Generated Interface**:
    -   Stateless logic functions: `generate(variant_id, new_data, state)`.
    -   Metadata structures for describing strategy inputs/outputs.

## Implementation Roadmap

1.  **Project Shell**: `cargo new comet`.
2.  **AST & Parser**:
    -   Implement structs from `ast.md`.
    -   Implement grammar from `parse.md` using `pest`.
3.  **Symbol Table**:
    -   Structs to hold `TypeInfo`, `BehaviorInfo`, `FunctionInfo`.
4.  **Synthesis Prototype**:
    -   Implement the "Context Branching" loop described in `transform.md` for a simple case (e.g., the Ratio vs Spread example).
5.  **Codegen**:
    -   Simple string-template based generation for rust.

## Verification

-   **Unit Tests**:
    -   Lexer/Parser tests for `lex.md` examples.
    -   AST serialization tests.
-   **Integration Tests**:
    -   End-to-end compilation of the `EventDrivenStrategy` example from `spec.md`.
    -   Verify that it generates the expected number of variants (e.g., 2 variants for `Comparator` * N variants for `days`).
