# Comet Language Specification

Comet is a domain-specific language for synthesize set of **functions** in quantitative finance. 
It formalizes logical structures.

## Design Philosophy

-   **Types** 
    - Keywords that represent data formats or semantic properties
    - Notation : `Series`, `DataFrame`, `Indicator`, `None`, `"EBIT"`, `"close"`
        - Quoted string is a data name.
    - Combination : `Series NonZero`
    - Assignment `type A : Series NonZero`

-   **Constraints**
    - Constraints is an expression that represents space of types.
    - Constraints can be expanded to a list of types. 
    - Constraints can be matched to determine valid types
    - Single type is a valid constraint.
    - Addition : (LHS) (RHS) 
        - When addition is expanded, RHS is appended to each type that matches LHS.
        - `Series Monetary` can be expanded to `[Series Monetary]`
        - `( Series | DataFrame NonZero | Monetary)` can be expanded to `[Series, DataFrame NonZero, Monetary]`
        - `( Series | DataFrame ) ( NonZero | Monetary)` can be expanded to `[Series NonZero, DataFrame NonZero, Series Monetary, DataFrame Monetary]`
        - None Type : Adding None type have no effect and removed when expanded.
            - `Series (None | NonZero) == Series | (Series NonZero)`
        - Same type added is removed when expanded : `( A A ) == A`
    - Union : (LHS) | (RHS)
        - When union is expanded, All type that matches RHS is appended to each type that matches LHS. Duplicates are removed.
        - `( A | B ) | ( C | D )` can be expanded to `[A, B, C, D]`
        - `A C | A C | A D` can be expanded to `[A C, A D]`
    - Subtract : (LHS) - (RHS) 
        - When subtraction is expanded, patterns that matches RHS is removed from LHS.
        - `( Series | DataFrame ) - DataFrame` can be expanded to `[Series]`
        - `( Series | DataFrame ) - (DataFrame NonZero)` can be expanded to `[Series, DataFrame]` because `DataFrame` is not matched by `DataFrame NonZero`.
        - `( Series | DataFrame ) NonZero - ( Series NonZero )` can be expanded to `[DataFrame NonZero]`.
    - Matching : 
        - Single type can be matched to a constraint, when expansion of the constraint includes the type.
            - type `Series NonZero` matches constraint `Series`, `NonZero`, `Series NonZero`  
        - Constraints can be matched to a constraint, when expansion of the constraint includes the constraint.
            - constraint `Series NonZero | DataFrame NonZero` matches constraint `NonZero`
            - constraint `Series NonZero | DataFrame NonZero | DataFrame` does not matches constraint `Nonzero` because "DataFrame" is not matched by `Nonzero`.
    - Assignment : 
        - Constraint variable '(symbol) can capture a constraints e.g. 'a , 'b  etc. 
        - Constraint can be stored to the constraint variable and recovered from the variable.

-   **Behaviors**
    - Behavior is a mapping from input type constraints to output type constraints.
    - To prevent infinite loop, behaviors are not allowed to be recursive, and only one function can be matched for each behavior.
    - `behavior Compare (signal: (DataFrame | Series) 'a, reference: (DataFrame | Series) Finite Positive) -> ('a Finite)`
        - Constraint variable 'a or 'b etc can be used to capture a type.
        - It means that Compare is a mapping from A, B into (a Finite). If A is DataFrame, the result is also DataFrame.
        - Behavior can be fully expanded into the following list : 
            - `Compare(signal=DataFrame, reference=DataFrame Finite Positive) -> DataFrame Finite`
            - `Compare(signal=Series, reference=Series Finite Positive) -> Series Finite`
        - These example functions with following types are valid for Compare:
            - `fn divide(signal:DataFrame, reference:DataFrame Finite Positive) -> DataFrame Finite `
            - `fn diff(signal:DataFrame Finite Positive SomeOtherType, reference:DataFrame Finite Positive SomeOtherType) -> DataFrame Finite `
            - `fn divide_1d(signal:Series, reference:Series Finite Positive) -> Series Finite `

-   **Flows**
    - Flow is a path of transformations from input type constraints to output type constraints.
        - Behavior with its parameters can be used to define a flow.
    - Flow can be defined by chaining functions, behaviors, other flows.
        - `flow volume_spike = Compare(signal=Volume, reference=HistoricalVolume(signal=Volume, lookback=days()))`
            - Given that `behavior days() -> Days ("21" | "63")` 
            - Volume is a flow, so parenthesis is not added.
            - Compare, HistoricalVolume, days are behaviors or functions, so parenthesis is added.
        - Flow can be matched to a chain of functions.
            - `rank_diff(signal=data("volume"), reference=ts_mean(signal=data("volume"), lookback=21))`
            - `rank_diff(signal=data("volume"), reference=ts_mean(signal=data("volume"), lookback=63))`
            - `divide(signal=data("volume"), reference=ts_mean(signal=data("volume"), lookback=21))`
            - `divide(signal=data("volume"), reference=ts_mean(signal=data("volume"), lookback=63))`
        - Each functions match each elements of chains, and keeps the composition structure of the chained behavior. 

-   **Functions**
    - Functions map that receives a list of concepts and returns a concept.
    - A function is a valid behavior.
    - `fn Ratio ( signal: DataFrame, reference: DataFrame Positive ) -> (DataFrame Finite) { return A / B }`
        - Input and output type with code segments. 
        - Type can be used define functions, but constraint cannot be used. 
    - Function can be matched to a behavior all of the following conditions are met: 
        - Input keywords are valid for the behavior.
        - Input types are valid for the behavior.
        - Output type is valid for the behavior.
    - Example: 
        - `fn Ratio ( signal: DataFrame, reference: DataFrame Positive ) -> (DataFrame Finite) { return A / B }` can be matched to a behavior `behavior Compare (signal: (DataFrame | Series) 'a, reference: (DataFrame | Series) 'b Finite Positive) -> ('a Finite)`
    - data/data.cm contains functions that returns data
        - `fn load_ebit() -> (DataFrame "EBIT"){ return ... }` \
        `fn load_ebitda() -> (DataFrame "EBITDA"){ return ... }`
        - These functions can be matched to a behavior `behavior earnings () -> DataFrame ("EBIT"|"EBITDA")`



## Syntax Overview

### 1. Terminology

To align with the "Semantic" design philosophy, we use specific terms:

| Concept | Description |
| :--- | :--- |
| **Type** | A label representing data format (`Series`) or semantic property (`NonZero`). |
| **Constraint** | An expression (Addition, Union, Subtraction) matching a valid set of Types. |
| **Behavior** | Abstract capability mapping input types to output types using constraints. |
| **Function** | A concrete logical implementation that can be synthesized if it matches a Behavior. |

### 2. File Structure

Comet supports modularity via imports.

```comet
import "stdlib.cm"
import "data/universe.cm"
```

### 3. Type Definitions

Types include data structures and semantic properties.

```comet
// Data Formats
type Series
type DataFrame

// Semantic Properties
type NonZero
type Monetary
type Stationary

// Derived Concepts
// (Using 'type' keyword for definition, although strictly 'Series NonZero' is a constraint)
type Indicator : DataFrame Stationary
type Volume : DataFrame NonZero   // Volume is a Series that is NonZero
type Price  : DataFrame Monetary  // Price is a Series that is Monetary
type Profit : DataFrame Monetary
type Days : Const NonZero Count
```

### 4. Behaviors & Functions

Behaviors define the "Interface" or "Trait" of an operation. Functions provide the "Implementation".

```comet
// Abstract Behavior
// Maps inputs A and B to an Indicator. B must be valid for the operations expected.
behavior Comparator(A: DataFrame, B: DataFrame) -> Indicator

// Function: Ratio
// Logic: A / B. Valid only if B is NonZero (to avoid division by zero).
fn ratio(signal: DataFrame, reference: DataFrame NonZero) -> Indicator {
    return signal / reference
}

// Function: Spread
// Logic: A - B. Valid generally for Series/DataFrames.
fn spread(A: DataFrame, B: DataFrame) -> Indicator {
    return A - B
}
```

## 5. Type-Driven Function Synthesis

The core of Comet is the **Flow**, where users describe a high-level intent, and the compiler synthesizes the valid mathematical operations based on the semantic properties.

### Complex Example: Event Driven Strategy

```comet
// 1. Inputs (Semantic Definition)
behavior price () -> DataFrame Monetary ("close"|"open")
behavior earnings () -> DataFrame Monetary ("EBIT"|"EBITDA")
behavior vol () -> DataFrame NonZero Count Volume ("volume")
behavior days () -> Const NonZero Count Days ("21", "63")
behavior levels () -> Const("0.1", "0.2")

// 2. Transformations
behavior HistoricalVolume(signal: DataFrame Volume, lookback: Const Days) -> DataFrame
fn ts_mean(x: DataFrame, days: Const Days) -> DataFrame { ... }
fn ts_delay(x: DataFrame, days: Const Days) -> DataFrame { ... }

flow volume_spike = Comparator(signal=vol(), reference=HistoricalVolume(signal=vol(), lookback=days()))

// 3. Trigger
fn trigger(signal: DataFrame, trigger: DataFrame, levels: Const) -> DataFrame Trigger {
    ...
    return signal 
}

flow result = trigger(signal=earnings(), trigger=volume_spike, levels=levels())
```

### Guarding Against Nonsense

If we tried to compare `Volume` and `0` using `Ratio`:
-   `0` does not generally have the `NonZero` property (unless explicitly typed).
-   Function `Ratio` requires `reference` to be `NonZero`.
-   Constraint violation -> **Ratio** is not generated.

This ensures that the generated strategies are **Categorically Consistent**.


### Synthesis 
- flow result is expanded.
- All possible combinations of implementation of behaviors are iterated. 
- Function types are checked and all valid combinations are synthesized. 