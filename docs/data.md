# Data Loading Interface (`data.md`)

This document defines how data is ingested into Comet strategies.

## 1. The `Universe` Generator

The `Universe` function is the primary entry point for data. It returns a signal of instruments as **Concrete Types**.

### Signature
`universe :: Type -> [Series Real]`

### Usage
```clean
// Fetch all instruments for the 'Earnings' type
// Returns a list of Series where each Series is a `Series Real`
// Attached with semantic properties (Instances) hidden in the context.
ebit = universe Earnings
```

## 2. Semantic Loading (Instances)

Data loaders automatically provide **Type Class Instances** for the loaded types. This enables the synthesis engine to know what can be done with the data.

-   **Volume**: Provides `instance NonZero Volume`, `instance Count Volume`.
-   **Price**: Provides `instance NonZero Price`, `instance Monetary Price`.
-   **Return**: Provides `instance Stationary Return`.

## 3. Time Alignment

Comet assumes an implicit time index. All data loaded via `Universe` is aligned to the master clock of the simulation.

-   `freq`: Property defined in the Type or metadata determining the native frequency.
