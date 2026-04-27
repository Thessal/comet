# RL State and Action definition

## Action Space

Action space is a union of 4 different types of action set.
### Action to Index Mapping (`action_to_id`)

Action implements methods for converting action from and into the token index.

`action_vocab_size = 1 (done) + 1 (reduce) + len(integers) + len(floats) + len(strings) + len(operators)`


## State Representation

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
