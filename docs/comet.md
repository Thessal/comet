## Comet

Comet is a domain-specific language for describing combinatorial designs in quantitative finance.

### Core Concepts

1.  **Universe**
    -   List of signals with same properties and structure
    -   For example, 'Earnings' universe is ['EBIT', 'Revenue', 'NetIncome'] with DataFrame structure and Monetary property

2.  **Properties (Semantic Type)**: 
    -   Represents a semantic abstraction defined by a set of **Properties**.
    -   These are used for logic dispatch and synthesis.

3.  **Structure (Data Container)**:
    -   Represents the underlying physical data layout.
    -   Example: `Series` (Time-series), `DataFrame` (Tabular), `Matrix` (3D Tensor).


