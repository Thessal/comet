# Data Loading Interface (`data.md`)

This document defines how data is ingested into Comet strategies.

## 1. The `Universe` Generator

The `Universe` function is the primary entry point for data.

### Signature
`fn Universe(type: Type) -> Series`

### Usage
```comet
// Fetch all instruments for the 'Earnings' type
ebit <- Universe(Earnings)
```

## 2. Semantic Loading

Data loaders automatically attach Semantic Properties based on the schema definition.

-   **Volume**: Loaded with `Count`, `NonZero`.
-   **Price**: Loaded with `Monetary`, `NonZero`.
-   **Return**: Loaded with `Stationary`.

## 3. Time Alignment

Comet assumes an implicit time index. All data loaded via `Universe` is aligned to the master clock of the simulation.

-   `freq`: Property defining the native frequency (Daily, Intraday).
-   `resample(x, freq)`: Explicit resampling.
