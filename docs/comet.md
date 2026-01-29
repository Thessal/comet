# Comet

Comet is a **Functional** domain-specific language for describing combinatorial designs in quantitative finance.

## Core Concepts

1.  **Universe (Generators)**
    -   Lazy lists of signals.
    -   Example: `universe Earnings` generates a stream of earnings data.

2.  **Semantic Properties (Type Classes)**:
    -   Properties like `NonZero`, `Stationary`, `Monetary` are defined as **Types Classes**.
    -   They define valid sets of operations (e.g., you can divide by something if it is `NonZero`).

3.  **Synthesis (Branching)**:
    -   Comet functions (Behaviors) can resolve to **multiple implementations**.
    -   The compiler iterates through all valid Type Class Instances for a given context, creating a product space of valid strategies.
