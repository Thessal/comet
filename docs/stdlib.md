# Standard Library & Built-in Operators

## Built-in Synthesis Logic

Some operators and behaviors are synthesized directly by the compiler engine without explicit `Implementation` blocks in the user code. This allows for complex type interactions (like broadcasting) that are difficult to express in the `Implementation` syntax.

### Binary Operators

#### Division (`/`)

The division operator is synthesized with strict type checking based on the operands' structural types and properties.

**Type Matrix:**

| Left Hand Side | Right Hand Side | Result Type | Notes |
| :--- | :--- | :--- | :--- |
| `DataFrame` | `DataFrame` | `DataFrame` | Element-wise division |
| `DataFrame` | `TimeSeries` | `DataFrame` | Broadcasting / Alignment |
| `DataFrame` | `Constant` | `DataFrame` | Scalar division |
| `TimeSeries` | `TimeSeries` | `TimeSeries` | Sparse alignment |
| `TimeSeries` | `Constant` | `TimeSeries` | Scalar division |
| `Constant` | `Constant` | `Constant` | Scalar calculation |

A type is considered `Constant` if it has the `Constant` property (e.g., `Integer`, `Float`).

**Example:**

```comet
vol_ratio = min_vol / hist_vol // TimeSeries / TimeSeries -> TimeSeries
```
