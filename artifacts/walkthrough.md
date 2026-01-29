# Clean-like Comet Syntax Exploration

## Overview
We explored redesigning Comet's syntax to align with the **Clean** functional programming language, transforming it from a procedural/imperative DSL to a functional DSL while maintaining its core "Synthesis" (multi-branch iteration) capability.

## Key Outcomes

### 1. Functional Mapping Established
We successfully mapped core Comet concepts to Functional equivalents:

| Comet Concept | Functional/Clean Concept | Reasoning |
| :--- | :--- | :--- |
| `Properties` | **Type Classes** | Constraints (`| Property a`) allow synthesized combinatorial logic without O(2^N) types. |
| `Structure` | **ADTs** | Standard algebraic types (`:: Series a = ...`). |
| `Behavior/Flow` | **Functions** | Multi-parameter type classes allow resolving multiple valid implementations for synthesis. |
| `Where` | **Constraints** | `where x is NonZero` -> `| NonZero x`. |
| `Ensures` | **Newtypes** | `ensures { Stationary }` -> Returns `Stationary a` wrapper. |

### 2. Synthesis Feasibility
We confirmed that "Branching" (Synthesis) can be implemented during the type resolution phase. When the compiler encounters a function call (e.g., `compare a b`), it can instantiate a subgraph for *every* valid Type Class instance, satisfying the "iterate all paths" requirement.

### 3. Syntax Proposal
A concrete syntax proposal was created and refinded in [implementation_plan.md](implementation_plan.md).

## Next Steps
*   **Prototype**: Implement a small parser/type-checker for this new syntax (likely in the `clean-language-report` folder or a new `comet-ng` folder).
*   **Compiler Update**: Refactor the synthesis engine to work on the new AST.
