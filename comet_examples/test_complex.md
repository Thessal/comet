# Concept Summary: `test_complex.cm`

This document summarizes the concepts used in the complex strategy example and outlines the structural requirements for the redesign.

## Concept Table

| Category | Name | Description | Used In Example |
| :--- | :--- | :--- | :--- |
| **Structure** | `Series` | Abstract time-series data (could be valid at sparse ticks). | Yes (Input/Output) |
| **Structure** | `DataFrame` | Aligned, dense tabular data (implicit in current design, explicit requirement). | Implicit |
| **Structure** | `Matrix` | Linear algebra structure (2D). | No (Future req) |
| **Combination** | `PERatio` | Domain-specific Instrument type (Financial Ratio). | Yes |
| **Combination** | `Volume` | Domain-specific Instrument type (Quantity). | Yes |
| **Behavior** | `Historical` | Logic to access past values or aggregate over a window. | Yes |
| **Behavior** | `Comparator` | Logic to compare two inputs (e.g., Ratio, Spread). | Yes |
| **Behavior** | `Normalizer` | Logic to standardize data (e.g., Z-Score). | Yes |
| **Impl** | `MovingAverage` | Implementation of `Historical`. | Yes |
| **Impl** | `Ratio` | Implementation of `Comparator`. | Yes |
| **Impl** | `ZScore` | Implementation of `Normalizer`. | Yes |
| **Impl** | `update_when` | Functional implementation to filter/update streams. | Yes |
| **Property** | `Monetary` | Semantic property indicating currency value. | Yes (PERatio) |
| **Property** | `NonZero` | Semantic property indicating non-zero values. | Yes (Volume) |
| **Property** | `Condition` | **Redefined**: Represents a state or filter property of a DataFrame (e.g., `x > 0.8`). | Yes (as Type) |

## Design Notes

1.  **Structures**: The current `Series` type is too generic. We need to distinguish between:
    -   `TimeSeries` (Sparse / Event-driven)
    -   `DataFrame` (Dense / Aligned / Table)
    -   `Matrix` (Math / Quant operations)

2.  **Condition as Property**:
    -   Instead of `Type Condition`, masking/filtering should be a property state of the `DataFrame`.
    -   Example: `DataFrame` has a property `Mask` or `Filter`.
    -   `greater_than(0.8)` acts as a modifier that attaches a `Condition` property to the structure.

3.  **Redesign Goal**:
    -   Update `test_complex.cm` to explicitly use `DataFrame` for the daily ratio and `TimeSeries` (or similar) for the minutely volume.
    -   Refactor `update_when` to respect the `Condition` property logic.
