# Standard Library & Built-in Operators

## Architecture

The `stdlib` defines operators and functions as a Rust library. When the `stdlib` codebase compiles, a dynamic library (`.so` file) is generated. This replaces the previous approach of distributing raw Rust source code files directly (e.g., `src/comet/functions`).

The functions within the `stdlib` are designed as stateful kernel functions (similar in application to `pd.DataFrame.rolling.apply`, but implemented natively). They must adhere to a strict structural convention:
- **Type Signatures**: Functions must have explicit input and output type signatures.
- **State Management**: Functions receive their stored internal state along with the incoming data, and they return the updated internal state and the outbound data.

## Built-in Synthesis Logic

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
