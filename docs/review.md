# Comet Documentation Review Findings

Here is a summary of the inconsistencies and incomplete instructions found across the `docs/` and `comet_examples/` directories:

## 1. Syntax & Grammar Inconsistencies (`parse.md` vs Examples)

- **Flow Blocks:** `parse.md` and `ast.md` define flows as single expressions (`flow Identifier = Expr`). However, `spec.md`, `test_basic.cm`, and `test_complex.cm` use a block syntax (`Flow Name { ... }`) with multiple statements and a `return` keyword.
- **Assignment Operator (`<-`):** The `.cm` examples extensively use the `<-` operator for intermediate variables (e.g., `price <- Universe(Price)`). This operator is completely missing from `parse.md` and `lex.md`, which only support `=`.
- **Implementation Keyword:** Examples extensively use `Implementation Name implements Behavior(x) { ... }`. Neither `parse.md`, `lex.md`, nor `ast.md` supports or defines an `Implementation` declaration block (they only specify `FuncDecl` via `fn`).
- **Type Derivation:** `stdlib.cm` uses `Type Integer : Any derives { Constant }`. However, `parse.md` defines types as `type Identifier (":" TypeRef*)?` and does not include the `derives { ... }` syntax.
- **Return Types in Signatures:** `spec.md` uses complex return constraints like `-> (DataFrame Finite)` or `-> ('a Finite)`. However, `parse.md` strictly restricts `FuncDecl` and `BehaviorDecl` return types to a single `TypeRef` (`Identifier`).
- **Keyword Case Sensitivity:** `lex.md` lists `Type`, `Behavior`, `Flow`, `Fn` as capitalized keywords. `parse.md` uses `"type"`, `"behavior"`, `"flow"`, `"fn"` in lowercase. The `.cm` examples mostly use the capitalized versions but `spec.md` sometimes mixes them.

## 2. Structural & AST Inconsistencies (`ast.md`)

- **Missing Enum Definition:** The `Declaration` enum in `ast.md` is missing its opening line (`pub enum Declaration {`) immediately after the `Program` struct.
- **Missing Nodes:** The AST lacks nodes to support the `Implementation` block and the block-based `Flow` structure with intermediate assignments seen in the examples.

## 3. Incomplete Instructions & TODOs

- **Memory/State Management (`compile.md`):** Needs a concrete design for state management during incremental updates: *"NOTE: we need to think about how to do the memory management. There are states for each operator, so passing it as a single blob might be tricky."*
- **Symbolic Regression (`compile.md`):** Missing implementation specifics: *"TODO: Expression of the strategy that can be used for the symbolic regression."* and *"IMPORTANT - We have to do causal analyais..."*
- **Terminology Update (`runtime.md`):** Needs refactor instruction executed: *"TODO: rename runtime into dataloader."*
- **Example Uncertainties (`test_basic.cm` & `test_complex.cm`):** Contain several inline comments indicating unresolved design decisions:
  - *"trim_ternary not in stdlib yet? assuming it exists or defined"*
  - *"Normalizer is Behavior. Can we call Behavior? ... Docs 'transform.md' So yes..."*
  - *"Logic for Unit check handled where? ... For now skipping implicit constraint or assuming it passes."*
  - *"Comparator needs to handle TimeSeries inputs now if we want that / Or we assume automatic promotion/alignment?"*
