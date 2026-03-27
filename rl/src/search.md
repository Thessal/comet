# Comet RL Action and Type Index Mapping Convention

This document outlines the embedding index assignments used to encode the current environment `SearchState` and sequence actions into uniform numeric vocabulary IDs. These numerical identifiers are mapped into the `TransformerModel`'s trainable embeddings during symbolic regression search.

## Action to Index Mapping (`action_to_id`)

The model's action vocabulary size dynamically scales alongside the explicitly configured constants (integers, floats, strings) and subset of operator macros extracted directly from the user's `BehaviorDecl`. 
The formula computes as:
`action_vocab_size = 2 + len(integers) + len(floats) + len(strings) + len(available_funcs)`

The IDs align to a continuous index scale mapping independently bounded sets:

- **0**: `Action::Done` 
  - Represents the concluding action step signaling successful expression termination reducing to the `target_return`.
- **1**: `Action::Shift`
  - Moves the next parameter requirement from the `unprocessed_params` pipeline into the operational evaluation context `stack`.
- **[Base Ints .. + len(integers)]**: `Action::ShiftInteger(v)`
  - `Base Ints` starts at `2`. Maps individual explicitly permitted integers defined within behavior context to a strictly unique index based on array position.
- **[Base Floats .. + len(floats)]**: `Action::ShiftFloat(v)`
  - Evaluates unique floating-point numbers provided in configurations (e.g. `[0.1, 0.5, 0.9]`), independently assigning `ID 2 + len(integers)` to `0.1`, matching each float to a singular categorical classification token.
- **[Base Strings .. + len(strings)]**: `Action::ShiftString(v)`
  - Same process applies exclusively for string literals uniquely isolated to specific parameter configurations (e.g. passing `"volume"` to an index class token distinctly separate from `"close"`).
- **[Base Funcs .. Max Size - 1]**: `Action::Reduce(identifier)`
  - Maps sequence closures down the array length corresponding explicitly to the alphabetically sorted function subsets available locally to evaluate across dimensions.

## Type to Index Mapping (`type_to_id`)

`TransformerModelConfig` requires a static parameter describing the embedding constraints bound for sequence states during translation, defined broadly as `type_vocab_size`.

Within the architecture, this is tightly scaled to boundary limit **8**. 
The configuration is established as exactly `1 + len(TypeDecl variants)`.
The `1` index securely reserves position `0` representing missing sequence elements or tensor padding, while `1-7` represent AST constants individually.

### Explicit Assignments 
| ID | Type / Constraint | Semantic Purpose |
|---|---|---|
| **0** | `PAD / UNKNOWN` | Represents empty slots in the tensor for unused stack allocations or missing sequence positions. |
| **1** | `TypeDecl::DataFrame` | |
| **2** | `TypeDecl::Matrix` | |
| **3** | `TypeDecl::Vector` | |
| **4** | `TypeDecl::String` | |
| **5** | `TypeDecl::Float` | |
| **6** | `TypeDecl::Bool` | |
| **7** | `TypeDecl::Void` | |
