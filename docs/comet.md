## Comet

Comet is a domain-specific language for describing combinatorial designs in quantitative finance.

### Core Concepts

1.  **Combination (Semantic Type)**: 
    -   Represents a semantic abstraction defined by a set of **Properties**.
    -   Example: `Type PERatio : Instrument derives { Monetary }`.
    -   These are used for logic dispatch and synthesis.

2.  **Structure (Data Container)**:
    -   Represents the underlying physical data layout.
    -   Example: `Series` (Time-series), `DataFrame` (Tabular), `Matrix` (2D Array).
    -   Combinations are *stored in* Structures.

### Syntax

XML based (Legacy reference, actual syntax is C-like/Rust-like).
