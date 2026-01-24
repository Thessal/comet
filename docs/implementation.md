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
    -   Register all `Type` definitions and their properties.
    -   Register all `Behavior` (Trait) signatures.
    -   Register all `Implementation` blocks.
    -   **Constraint Verification**: Validate that `where` clauses used in definitions are semantically valid (e.g., checking properties that exist).

### 3. Synthesis Engine (The Core)
-   **Goal**: Expand `Flow` definitions into concrete Execution Trees.
-   **Reference**: `transform.md`.
-   **Key Logic**:
    -   **Context Management**: Tracking active `Variable -> Value` bindings and their Semantic Properties.
    -   **Branching**: Splitting contexts when multiple Implementations match.
    -   **Pruning**: Dropping contexts where Constraints (`where`) fail.

### 4. Backend (Code Generation)
-   **Goal**: Emit executable code for the Target Runtime.
-   **Reference**: `runtime.md`.
-   **Targets**:
    -   **Python**: Generate a `.py` script using a library like `pandas`.
    -   **Rust**: Generate a `.rs` file using `polars` or similar.

## Implementation Roadmap

1.  **Project Shell**: `cargo new comet`.
2.  **AST & Parser**:
    -   Implement structs from `ast.md`.
    -   Implement grammar from `parse.md` using `pest`.
3.  **Symbol Table**:
    -   Structs to hold `TypeInfo`, `BehaviorInfo`, `ImplInfo`.
4.  **Synthesis Prototype**:
    -   Implement the "Context Branching" loop described in `transform.md` for a simple case (e.g., the Ratio vs Spread example).
5.  **Codegen**:
    -   Simple string-template based generation for Python.

## Verification

-   **Unit Tests**:
    -   Lexer/Parser tests for `lex.md` examples.
    -   AST serialization tests.
-   **Integration Tests**:
    -   End-to-end compilation of the `EventDrivenStrategy` example from `spec.md`.
    -   Verify that it generates the expected number of variants (e.g., 2 variants for `Comparator` * N variants for `days`).
