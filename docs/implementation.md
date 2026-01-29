# Comet Compiler Implementation Plan

This document outlines the plan to build the Comet compiler.

**Reference Documents**:
-   [Language Spec](spec.md)
-   [AST Spec](ast.md)
-   [Synthesis Logic](transform.md)
-   [Compile Strategy](compile.md)

## Architecture

### 1. Frontend (Lexing & Parsing)
-   **Goal**: Transform Clean-like source into the `Program` AST.
-   **Syntax**: Follows the [Clean Language Report](../clean-language-report/doc/CleanLanguageReport.html).

### 2. Semantic Analysis (Type Checking)
-   **Goal**: Populate the Symbol Table and resolving Types.
-   **Key Components**:
    -   **Class Registry**: Stores `class` definitions.
    -   **Instance Registry**: Stores `instance` definitions (Global Facts).
    -   **Type Checker**: Standard Hindley-Milner inference adapted for multi-parameter type classes.

### 3. Synthesis Engine (The Core)
-   **Goal**: Expand functional flows into Execution Trees.
-   **Key Logic**:
    -   **Instance Resolution**: Finding all matching instances for a constraint `Comparator a b c`.
    -   **Combinatorial Expansion**: Branching the graph for every valid instance found.
    -   **Constraint Verification**: Pruning branches where constraints (`| Predicate`) fail.

### 4. Backend (Code Generation)
-   **Strategy**: **Transpilation to Rust**.
-   **Process**:
    -   Each valid execution tree is emitted as a Rust function/struct.
    -   Common logic is shared via the standard library.

## Implementation Roadmap

1.  **Project Shell**: `cargo new comet`.
2.  **AST**: Implement `ast.md`.
3.  **Parser**: Implement a subset of Clean syntax (Modules, Classes, Instances, Functions).
4.  **Resolver**: Implement Instance Resolution logic.
5.  **Codegen**: Emit Rust code.

## Verification

-   **Unit Tests**: Parser tests for Clean syntax.
-   **Integration**: Compile `EventDrivenStrategy` and verify correct number of generated variants.
