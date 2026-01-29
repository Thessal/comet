# Clean-like Comet Syntax Proposal

## Goal
Redesign the Comet language syntax to align with the "Clean" functional programming style, moving from a procedural/imperative model to a pure functional model with Types, Guards, and Functions, while managing the unique "Synthesis" requirement (iterating all valid branches).

## User Review Required
> [!IMPORTANT]
> **Paradigm Shift**: This proposal changes Comet from an imperative-style DSL (`v <- Comparator(a,b)`) to a functional-style DSL (`v = comparator a b`).
> **Synthesis in Functional Context**: Standard functional languages resolve to *one* function. Comet must resolve to *many*. We propose using **Overlapping Instances** or **Multi-Resolution** on Type Classes to achieve this.

## Proposed Syntax Changes

### 1. File Structure & Modules
**Old**: Implicit modules, C-style imports.
**New**: Clean-style modules.

```clean
module Strategy
import StdEnv
```

### 2. Types & Structures
**Old**: `Struct`, `Type ... derives ...`
**New**: Algebraic Data Types (ADTs) & Type Synonyms.

```clean
// Container
:: Series a = Series a

// Domain Types
:: Volume :== Series Real
:: Price  :== Series Real
```

### 3. Properties -> Type Classes & Constraints
**Old**: `Property NonZero`, `Type Volume ... derives { NonZero }`
**New**: Type Classes as Constraints (+ Newtypes for added properties).

#### Efficient Handling of Combinatorial Types
The user notes that 2 properties (`Profitability`, `Liquidity`) on 2 structures (`TimeSeries`, `DataFrame`) could conceptually create 8 types.
In Clean/Haskell, we avoid generating 8 distinct structs. Instead, we use **Constraints**.

*   **Data Definition**: Only 2 base types.
    ```clean
    :: TimeSeries = ...
    :: DataFrame = ...
    ```
*   **Property Definition**: properties are Type Classes.
    ```clean
    class Profitability a
    class Liquidity a
    
    // Base instances
    instance Profitability TimeSeries
    instance Liquidity DataFrame
    ```
*   **Combinatorial Usage**: We don't create a `ProfitabilityLiquidityTimeSeries` type. We write functions that demand both:
    ```clean
    myFunction :: a -> Result | Profitability a & Liquidity a
    ```
    The compiler passes the necessary "dictionaries" (evidences) for both valid combinations. If a type has both, it works. If not, it fails. This is O(P) (number of properties) rather than O(2^P) (combinatorial explosion of types).

#### "Adding" Properties (Ensures) via Newtypes
If a function *adds* a property (e.g., `Stationary` after differencing), we use Newtype Wrappers:
```clean
:: Stationary a = Stationary a // Wrapper
instance Stationary (Stationary a) // It definitely has the property
```
Transformation: `diff :: a -> Stationary a`.

### 4. Semantics/Behaviors -> Type Classes
**Old**: `Behavior Comparator(a, b) -> Res`
**New**: Multi-parameter Type Classes.

### 5. "Where" and "Ensures" Mapping
These strict Comet concepts map directly to Functional concepts:

| Comet Concept | Purpose | Functional/Clean Equivalent | Example |
| :--- | :--- | :--- | :--- |
| **Where** | Pre-condition on Input | **Type Class Constraints** (`|`) | `matches ... where x is NonZero` <br> `instance ... | NonZero x` |
| **Ensures** | Post-condition on Output | **Return Type Wrappers** | `ensures { Stationary }` <br> `-> Stationary ResultType` |

**Example of Where/Ensures:**

**Old Comet**:
```comet
Implementation Diff implements Transformation(x)
where x is NonZero       // Input constraint
ensures { Stationary }   // Output guarantee
{ return x.diff() }
```

**New Clean-Comet**:
```clean
// 1. "Where" -> Constraint (| NonZero x)
// 2. "Ensures" -> Return Type (Stationary x)
instance Transformation x (Stationary x) | NonZero x
    where transform x = Stationary (diff x)
```


### 5. Implementations -> Instances
**Old**: `Implementation Ratio ... matches Comparator`
**New**: Instances of the class.

> [!NOTE]
> **Core Difference**: In standard Clean, having multiple matching instances is an error. In Comet, this is the trigger for **Branching**.

```clean
// Ratio Logic
instance Comparator Volume Volume Series | NonZero a
    where compare v1 v2 = v1 / v2

// Spread Logic
instance Comparator Volume Volume Series | SameUnit v1 v2
    where compare v1 v2 = v1 - v2
```

### 6. Flow -> Function Definition
**Old**: `Flow Name { ... }` with `<-` assignment.
**New**: Standard function definition with `let ... in` or `where` blocks.

```clean
strategy :: Volume Volume -> Signal
strategy v1 v2 = 
    let 
        // "compare" triggers synthesis of ALL valid instances (Ratio, Spread)
        cmp = compare v1 v2 
        norm = normalize cmp
    in 
        trigger norm
```

## Comparisons

### Example: Complex Strategy ([test_complex.cm](file:///home/jongkook90/antigravity/comet/comet_examples/test_complex.cm))

#### Old Comet
```comet
Flow Strategy {
    daily_pe <- PE_Universe
    min_vol  <- Volume_Universe
    hist_vol <- Historical(min_vol)
    vol_ratio <- Comparator(min_vol, hist_vol)
    spike_score <- Normalizer(vol_ratio)
    result = update_when(daily_pe, spike_score)
    return result
}
```

#### New "Clean" Comet
```clean
module Strategy

import StdLib
import Data

// Types
:: StrategyResult :== Series Bool

// The Strategy Function
strategy :: Universe PE -> Universe Volume -> StrategyResult
strategy daily_pe min_vol =
    let
        // Synthesis: "historical" finds all valid implementations (e.g., MovingAverage)
        hist_vol = historical min_vol
        
        // Synthesis: "compare" finds Ratio, Spread, etc.
        // Note: min_vol and hist_vol must be compatible types
        vol_ratio = compare min_vol hist_vol
        
        // Synthesis: "normalize" finds ZScore, Rank, etc.
        spike_score = normalize vol_ratio
        
        // Explicit logic
        masked_spike = filter spike_score 0.8
    in
        updateWhen daily_pe masked_spike
```

## Synthesis Feasibility
The core requirement is "iterate through all possible branches".
In the Functional model:
1.  **Nodes** are function calls (`compare`).
2.  **Edges** are data dependencies (`min_vol -> compare`).
3.  **Branching**: When the compiler encounters `compare v1 v2`:
    *   It looks up the `Comparator` class.
    *   It finds ALL instances matching the types of `v1`, `v2`.
    *   It checks the Guards (`| condition`).
    *   It instantiates a separate "Strategy Graph" for *each* passing instance.

This effectively turns the "Function Call" into a "Non-Deterministic Choice Points" during compilation (Synthesis).

## Verification Plan
Since this is a syntax proposal, verification involves:
1.  Reviewing with the User (You).
2.  Ensuring the syntax can express all features of [test_complex.cm](file:///home/jongkook90/antigravity/comet/comet_examples/test_complex.cm).
3.  Confirming the "Branching" logic is sound within the type system proposal.

I will request your approval on this text before implementing any changes (if implementation was requested, but currently the task is exploration/proposal).
