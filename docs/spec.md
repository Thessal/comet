# Comet Language Specification

Comet is a domain-specific language for synthesize set of **functions** in quantitative finance. 
It formalizes logical structures.

## Design Philosophy

-   **Types** 
    - Kewords that represent data formats or semantic properties
    - Notation : `Series`, `DataFrame`, `Indicator`, `None`
    - Combination : `Series NonZero`
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
    - Union : (LHS) | (RHS)
        - When union is expanded, All type that matches RHS is appended to each type that matches LHS.
        - `( A | B ) | ( C | D )` can be expanded to `[A, B, C, D]`
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
    - Behavior is a **Type Class** that maps input type constraints to output type constraints.
    - `Behavior Compare (dividend: (DataFrame | Series) 'a, divisor: (DataFrame | Series) Finite Positive) -> ('a Finite)`
        - Constraint variable 'a or 'b etc can be used to capture a type.
        - It means that Compare is a mapping from A, B into (a Finite). If A is DataFrame, the result is also DataFrame.
        - Behavior can be fully expanded into the following list : 
            - `Compare(dividend=DataFrame, divisor=DataFrame Finite Positive) -> DataFrame Finite`
            - `Compare(dividend=Series, divisor=Series Finite Positive) -> Series Finite`
        - These example functions with following types are valid for Compare:
            - fn divide(dividend:DataFrame, divisor:DataFrame Finite Positive) -> DataFrame Finite 
            - fn diff(dividend:DataFrame Finite Positive SomeOtherType, divisor:DataFrame Finite Positive SomeOtherType) -> DataFrame Finite 
            - fn divide_1d(dividend:Series, divisor:Series Finite Positive) -> Series Finite 
    - Behaviors can be chained and assigned to a concept.
    - Example: \
      `Behavior RemoveNegative (A: (DataFrame | Series) a) -> (a Positive)` \
      `Behavior RemoveZero (A: (DataFrame | Series) a) -> (a Positive)` \
      `price_safe = RemoveZero(A=RemoveNegative(A=price)) // assignment of chained behavior into a concept.` \
      `price_safe_2 = RemoveZero(A=RemoveNegative(A=price))` \
      `comparison_result = Compare(dividend=price_safe, divisor=price_safe_2) // Using saved concept.`    

-   **Functions**
    - Functions map that receives a list of concepts and returns a concept.
    - `fn Ratio ( dividend: DataFrame, divisor: DataFrame Positive ) -> (DataFrame Finite) { return A / B }`
        - Input and output type with code segments. 
        - Type can be used define functions, but constraint cannot be used. 
    - Function can be matched to a behavior all of the following conditions are met: 
        - Input keywords are valid for the behavior.
        - Input types are valid for the behavior.
        - Output type is valid for the behavior.
    - Example: 
        - `fn Ratio ( dividend: DataFrame, divisor: DataFrame Positive ) -> (DataFrame Finite) { return A / B }` can be matched to a behavior `Behavior Compare (dividend: (DataFrame | Series) 'a, divisor: (DataFrame | Series) 'b Finite Positive) -> ('a Finite)`



## Syntax Overview

### 1. Terminology: Quant vs. Code

To align with the "Semantic" design philosophy, we use specific terms:

<!-- | Concept | Replaces (CS Term) | Description |
| :--- | :--- | :--- |
| **Type** | `Constraint` / `Marker` | A semantic label (e.g., `NonZero`, `USD`). It describes the *meaning* of the data, enforcing categorical consistency. |
| **Property** | `Constraint` / `Marker` | A semantic label (e.g., `NonZero`, `USD`). It describes the *meaning* of the data, enforcing categorical consistency. |
| **Behavior** | `Class` | Defines an abstract logical capability (e.g., `Comparator`). It is about *what* makes sense to do with data, not just what methods exist. |
| **Logic** | `Instance` | A specific, grounded logic for a Behavior (e.g., `Ratio`, `Spread`). It represents a *valid hypothesis* or *model* for that behavior. | -->

### 2. File Structure

Comet supports modularity via imports.

```comet
import "stdlib.cm"
import "data/universe.cm"
```
<!-- 
### 3. Semantic Attributes (Properties)

We define logic properties that can be attached to data.

```comet
// Logical Markers
Property NonZero
Property Stationary
Property Count
Property Monetary
```

### 2. Domain Types

Types are defined by their structure AND their semantic properties.

```comet
// Basic Representation (The "Container")
Struct Series {}

// Domain Concepts (The "Meaning")
Type Volume : Series derives { NonZero, Count }
Type Price  : Series derives { NonZero, Monetary }
Type Return : Series derives { Stationary }
```

### 3. Behaviors (Traits) & Logic

Behaviors are defined on **abstract concepts**, not just specific structs.

```comet
// Abstract Behavior
Behavior Comparator(A, B) -> Indicator

// Logic: Ratio
// "A Ratio comparison is valid if B is physically capable of being a denominator (NonZero)"
Implementation Ratio implements Comparator(A, B) 
where B is NonZero 
{
    return A / B
}

// Logic: Spread
// "A Spread comparison is valid if A and B share the same Unit"
Implementation Spread implements Comparator(A, B)
where A.Unit == B.Unit
{
    return A - B
}
```

## 5. Type-Driven Function Synthesis

The core of Comet is the **Flow**, where users describe a high-level intent, and the compiler synthesizes the valid mathematical operations based on the semantic properties.

### Complex Example: Event Driven Strategy

```comet
Flow EventDrivenStrategy {
    // 1. Inputs (Semantic Definition)
    // "Get me US Earnings and Volume data"
    ebit <- Universe(Earnings) where Unit is USD
    vol  <- Universe(Volume)   // inherently NonZero, Count
    
    // 2. Transformations
    days <- [21, 63]
    avg_vol <- MovingAverage(vol, days) // Inherits 'Count' and 'NonZero' from vol? Yes.

    // 3. Synthesis (The "Quant" Logic)
    // We request a Comparison. The compiler checks available Implementations.
    
    // Check: Can we use "Ratio"?
    // Logic: Is 'avg_vol' NonZero? 
    // Yes (derived from Volume). -> Valid.
    
    // Check: Can we use "Spread"?
    // Logic: Do 'vol' and 'avg_vol' share units?
    // Yes (both are Count). -> Valid.
    
    // Result: Both Ratio and Spread are generated.
    spike <- Comparator(vol, avg_vol)

    // 4. Trigger
    signal <- Trigger(ebit, spike, Condition.GreaterThan, 4.0)
    
    return signal
}
```

### Guarding Against Nonsense

If we tried to compare `Volume` and `Price` using `Spread`:
-   `Volume` is `Count`
-   `Price` is `Monetary`
-   Condition `A.Unit == B.Unit` fails.
-> **Spread** is not generated.

If we tried to compare `Volume` and `0` using `Ratio`:
-   `0` is not `NonZero`.
-   Condition `B is NonZero` fails.
-> **Ratio** is not generated.

This ensures that the generated strategies are **Categorically Consistent**. -->
